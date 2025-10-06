from tqdm import tqdm
import time
from torch.nn.functional import softmax
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from codes.utils.functions import setup_parser
from codes.model.lightning_demel import LightningForDEMEL
from codes.utils.dataset import DataModuleForMEL


class ValidationErrorAnalysisCallback(Callback):
    def __init__(self, output_dir="/root/valid_before_test", batch_size=32, device="cuda:0"):  # todo you need to modify this
        super().__init__()
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.preds = []
        self.targets = []
        self.logits = []
        self.sample_ids = []
        self.entropy = None

    def on_test_start(self, trainer: Trainer, pl_module):
        os.makedirs(self.output_dir, exist_ok=True)
        pl_module.to(self.device)
        val_dataloader = trainer.datamodule.val_dataloader()

        pl_module.eval()
        pl_module.on_validation_start()
        outputs_list = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                batch.data = {k: v.to(self.device) for k, v in batch.data.items()}
                outputs = pl_module.validation_step(batch, 0)
                outputs_list.append(outputs)
                self.logits.append(outputs["logits"].detach())
                self.preds.append(torch.argmax(outputs["logits"], dim=1).detach())
                self.targets.append(outputs["targets"].detach())

        pl_module.validation_epoch_end(outputs_list)
        self.logits = torch.cat(self.logits, dim=0)
        self.preds = torch.cat(self.preds, dim=0)
        self.targets = torch.cat(self.targets, dim=0)

        error_mask = self.preds != self.targets
        error_indices = torch.where(error_mask)[0].cpu().numpy()
        self.entropy = -torch.sum(softmax(self.logits, dim=1) * torch.log(softmax(self.logits, dim=1) + 1e-10), dim=1)
        # 定长窗口
        entropy_range = self.find_optimal_entropy_range(error_indices, len(error_indices))
        pl_module.entropy_l_thresh = entropy_range[0]
        pl_module.entropy_h_thresh = entropy_range[1]

        del self.logits, self.preds, self.targets
        torch.cuda.empty_cache()

    def find_optimal_entropy_range(self, misclassified_indices, min_samples):
        # 创建错分样本掩码
        misclassified_mask = torch.zeros_like(self.entropy, dtype=torch.bool)
        misclassified_mask[misclassified_indices] = True
        # 按熵值排序
        sorted_entropies, sorted_indices = torch.sort(self.entropy)
        sorted_misclassified_mask = misclassified_mask[sorted_indices]

        n = len(sorted_entropies)
        best_ratio = 0
        best_range = (0, 0)
        best_count = 0

        # 滑动窗口遍历所有可能的区间
        for i in tqdm(range(n)):
            for j in range(i + min_samples, n + 1):  # 确保区间内至少有min_samples个样本
                # 计算区间内的样本数和错分样本数
                total_count = j - i
                misclassified_count = sorted_misclassified_mask[i:j].sum().item()
                if total_count > 0:
                    ratio = misclassified_count / total_count
                    # 更新最佳区间
                    if ratio > best_ratio or (ratio == best_ratio and total_count > best_count):
                        best_ratio = ratio
                        best_range = (sorted_entropies[i].item(), sorted_entropies[j - 1].item())
                        best_count = total_count

        return best_range[0] / torch.max(sorted_entropies), best_range[1] / torch.max(sorted_entropies), best_ratio, best_count


if __name__ == '__main__':
    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModuleForMEL(args)
    lightning_model = LightningForDEMEL(args)

    logger = pl.loggers.CSVLogger("./runs", name=args.run_name, flush_logs_every_n_steps=30)

    callbacks_list = []
    ckpt_callbacks = ModelCheckpoint(monitor='Val/mrr', save_weights_only=True, mode='max')
    early_stop_callback = EarlyStopping(monitor="Val/mrr", min_delta=0.00, patience=5, verbose=True, mode="max")
    callbacks_list.append(ckpt_callbacks)
    callbacks_list.append(early_stop_callback)

    if args.rerank:
        error_analysis_callback = ValidationErrorAnalysisCallback(
            output_dir="/root/valid_before_test",  # todo you need to modify this
            device="cuda:0")
        callbacks_list.append(error_analysis_callback)

    trainer = pl.Trainer(**args.trainer,
                         deterministic=True, logger=logger, default_root_dir="./runs",
                         callbacks=callbacks_list)

    trainer.fit(lightning_model, datamodule=data_module)
    trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')
