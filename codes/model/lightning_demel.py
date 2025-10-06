import json
import os
import pickle
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax
import pytorch_lightning as pl
from tqdm import tqdm
from codes.model.modeling_mel import MEL_Encoder, MEL_Matcher
from codes.utils.functions import *


class LightningForDEMEL(pl.LightningModule):
    def __init__(self, args):
        super(LightningForDEMEL, self).__init__()
        self.ckpt_matcher = None
        self.ckpt_encoder = None
        self.args = args
        self.save_hyperparameters(args)

        self.encoder = MEL_Encoder(args)
        self.matcher = MEL_Matcher(args)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.epoch_index = 0
        self.batch_index = 0

        self.temperature = 4
        self.alpha = 2.25
        self.entropy_l_thresh = None
        self.entropy_h_thresh = None
        self.topk = 10
        self.logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
        self.fuzzy_count = 0

    def on_train_epoch_start(self) -> None:
        ckpt_dir = self.logger.log_dir + '/checkpoints'
        if os.path.exists(ckpt_dir):
            self.ckpt_encoder = self.load_from_checkpoint(ckpt_dir + '/' + os.listdir(ckpt_dir)[0],
                                                          strict=False).encoder.to(torch.device('cuda:0'))
            self.ckpt_matcher = self.load_from_checkpoint(ckpt_dir + '/' + os.listdir(ckpt_dir)[0],
                                                          strict=False).matcher.to(torch.device('cuda:0'))

    def training_step(self, batch):
        ent_batch = {}
        mention_batch = {}
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v
        entity_empty_image_flag = ent_batch.pop('empty_img_flag')

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_batch)
        entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**ent_batch)

        logits, (text_logits, image_logits, image_text_logits) = self.matcher(entity_text_embeds,
                                                                              entity_text_seq_tokens,
                                                                              mention_text_embeds,
                                                                              mention_text_seq_tokens,
                                                                              entity_image_embeds,
                                                                              entity_image_patch_tokens,
                                                                              mention_image_embeds,
                                                                              mention_image_patch_tokens)
        labels = torch.arange(len(mention_text_embeds)).long().to(mention_text_embeds.device)

        if self.epoch_index == 0:
            text_loss = self.loss_fct(text_logits / self.temperature, labels)
            image_loss = self.loss_fct(image_logits / self.temperature, labels)
            image_text_loss = self.loss_fct(image_text_logits / self.temperature, labels)
            overall_loss = self.loss_fct(logits / self.temperature, labels)
            with open(self.args.data.batch_file_folder + f'/batch_{str(self.batch_index)}.pkl', 'wb') as f:
                pickle.dump(batch.data, f)
        # 存在checkpoint
        elif self.ckpt_encoder is not None and self.ckpt_matcher is not None:
            ckpt_mention_text_embeds, ckpt_mention_image_embeds, ckpt_mention_text_seq_tokens, ckpt_mention_image_patch_tokens = \
                self.ckpt_encoder(**mention_batch)
            ckpt_entity_text_embeds, ckpt_entity_image_embeds, ckpt_entity_text_seq_tokens, ckpt_entity_image_patch_tokens = \
                self.ckpt_encoder(**ent_batch)

            # 教师的常温logits
            ckpt_logits, (ckpt_text_logits, ckpt_image_logits, ckpt_image_text_logits) = self.ckpt_matcher(
                ckpt_entity_text_embeds,
                ckpt_entity_text_seq_tokens,
                ckpt_mention_text_embeds,
                ckpt_mention_text_seq_tokens,
                ckpt_entity_image_embeds,
                ckpt_entity_image_patch_tokens,
                ckpt_mention_image_embeds,
                ckpt_mention_image_patch_tokens)

            # 教师的高温软标签
            ckpt_labels_overall = self.compute_teacher_soft_labels(ckpt_logits)
            ckpt_labels_text = self.compute_teacher_soft_labels(ckpt_text_logits)
            ckpt_labels_image = self.compute_teacher_soft_labels(ckpt_image_logits)
            ckpt_labels_image_text = self.compute_teacher_soft_labels(ckpt_image_text_logits)

            loss_ckpt_text = self.compute_loss(ckpt_text_logits, labels)
            loss_ckpt_image = self.compute_loss(ckpt_image_logits, labels)
            loss_ckpt_image_text = self.compute_loss(ckpt_image_text_logits, labels)
            loss_ckpt_overall = self.compute_loss(ckpt_logits, labels)

            text_loss = self.compute_weighted_loss(text_logits, ckpt_labels_text,
                                                   loss_ckpt_text, labels)
            image_loss = self.compute_weighted_loss(image_logits, ckpt_labels_image,
                                                    loss_ckpt_image, labels)
            image_text_loss = self.compute_weighted_loss(image_text_logits, ckpt_labels_image_text,
                                                         loss_ckpt_image_text, labels)
            overall_loss = self.compute_weighted_loss(logits, ckpt_labels_overall,
                                                      loss_ckpt_overall, labels)
        else:
            with open(self.args.data.soft_labels_folder + f'/soft_label_{str(self.batch_index)}.pkl', 'rb') as f:
                soft_labels_overall, soft_labels_text, soft_labels_image, soft_labels_image_text, loss_t_overall, loss_t_text, loss_t_image, loss_t_image_text = pickle.load(f)

            text_loss = self.compute_weighted_loss(text_logits, soft_labels_text, loss_t_text, labels)
            image_loss = self.compute_weighted_loss(image_logits, soft_labels_image, loss_t_image, labels)
            image_text_loss = self.compute_weighted_loss(image_text_logits, soft_labels_image_text,
                                                         loss_t_image_text, labels)
            overall_loss = self.compute_weighted_loss(logits, soft_labels_overall, loss_t_overall, labels)

        loss = overall_loss + text_loss + image_loss + image_text_loss
        self.log('Train/loss', loss.detach().cpu().item(), on_epoch=True, prog_bar=True)
        self.batch_index += 1
        torch.cuda.empty_cache()
        return loss

    def training_epoch_end(self, outputs) -> None:
        if not os.path.exists(self.logger.log_dir + '/checkpoints'):
            for _ in range(self.batch_index):
                with open(self.args.data.batch_file_folder + f'/batch_{str(_)}.pkl', 'rb') as f:
                    batch_data = pickle.load(f)

                ent_batch = {}
                mention_batch = {}
                for k, v in batch_data.items():
                    if k.startswith('ent_'):
                        ent_batch[k.replace('ent_', '')] = v
                    else:
                        mention_batch[k] = v
                entity_empty_image_flag = ent_batch.pop('empty_img_flag')
                del batch_data
                mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
                    self.encoder(**mention_batch)
                entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
                    self.encoder(**ent_batch)
                labels = torch.arange(len(mention_text_embeds)).long().to(mention_text_embeds.device)
                # 老师的常温logits
                logits, (text_logits, image_logits, image_text_logits) = self.matcher(entity_text_embeds,
                                                                                      entity_text_seq_tokens,
                                                                                      mention_text_embeds,
                                                                                      mention_text_seq_tokens,
                                                                                      entity_image_embeds,
                                                                                      entity_image_patch_tokens,
                                                                                      mention_image_embeds,
                                                                                      mention_image_patch_tokens)
                soft_labels_overall = self.compute_teacher_soft_labels(logits)
                soft_labels_text = self.compute_teacher_soft_labels(text_logits)
                soft_labels_image = self.compute_teacher_soft_labels(image_logits)
                soft_labels_image_text = self.compute_teacher_soft_labels(image_text_logits)

                loss_t_overall = self.compute_loss(logits, labels)
                loss_t_text = self.compute_loss(text_logits, labels)
                loss_t_image = self.compute_loss(image_logits, labels)
                loss_t_image_text = self.compute_loss(image_text_logits, labels)
                with open(self.args.data.soft_labels_folder + f'/soft_label_{str(_)}.pkl', 'wb') as f:
                    pickle.dump((
                        soft_labels_overall, soft_labels_text, soft_labels_image, soft_labels_image_text,
                        loss_t_overall, loss_t_text, loss_t_image, loss_t_image_text),
                        f)

        self.batch_index = 0
        self.epoch_index += 1

    def on_validation_start(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        outputs_text_embed = []
        outputs_image_embed = []
        outputs_text_seq_tokens = []
        outputs_image_patch_tokens = []

        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='UpdateEmbed', total=len(entity_dataloader), dynamic_ncols=True):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
                    self.encoder(**batch)
                outputs_text_embed.append(entity_text_embeds.cpu())
                outputs_image_embed.append(entity_image_embeds.cpu())
                outputs_text_seq_tokens.append(entity_text_seq_tokens.cpu())
                outputs_image_patch_tokens.append(entity_image_patch_tokens.cpu())

        self.entity_text_embeds = torch.concat(outputs_text_embed, dim=0)
        self.entity_image_embeds = torch.concat(outputs_image_embed, dim=0)
        self.entity_text_seq_tokens = torch.concat(outputs_text_seq_tokens, dim=0)
        self.entity_image_patch_tokens = torch.concat(outputs_image_patch_tokens, dim=0)

    def validation_step(self, batch, batch_idx):
        answer = batch.pop('answer')
        batch_size = len(answer)
        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**batch)

        scores = []
        chunk_size = self.args.data.eval_chunk_size
        for idx in range(math.ceil(self.args.data.num_entity / chunk_size)):
            start_pos = idx * chunk_size
            end_pos = (idx + 1) * chunk_size

            chunk_entity_text_embeds = self.entity_text_embeds[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_image_embeds = self.entity_image_embeds[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_text_seq_tokens = self.entity_text_seq_tokens[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_image_patch_tokens = self.entity_image_patch_tokens[start_pos:end_pos].to(
                mention_text_embeds.device)

            chunk_score, _ = self.matcher(chunk_entity_text_embeds, chunk_entity_text_seq_tokens,
                                          mention_text_embeds, mention_text_seq_tokens,
                                          chunk_entity_image_embeds, chunk_entity_image_patch_tokens,
                                          mention_image_embeds, mention_image_patch_tokens)
            scores.append(chunk_score)

        scores = torch.concat(scores, dim=-1)
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()
        return dict(rank=tgt_rank, all_rank=rank.detach().cpu().numpy(), logits=scores, targets=answer)

    def validation_epoch_end(self, outputs):
        self.entity_text_embeds = None
        self.entity_image_embeds = None
        self.entity_text_seq_tokens = None
        self.entity_image_patch_tokens = None

        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Val/hits20", hits20)
        self.log("Val/hits10", hits10)
        self.log("Val/hits5", hits5)
        self.log("Val/hits3", hits3)
        self.log("Val/hits1", hits1)
        self.log("Val/mr", ranks.mean())
        self.log("Val/mrr", (1. / ranks).mean())

    def on_test_start(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()
        outputs_text_embed = []
        outputs_image_embed = []
        outputs_text_seq_tokens = []
        outputs_image_patch_tokens = []

        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='UpdateEmbed', total=len(entity_dataloader), dynamic_ncols=True):
                batch = pl.utilities.move_data_to_device(batch, self.device)
                entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
                    self.encoder(**batch)
                outputs_text_embed.append(entity_text_embeds.cpu())
                outputs_image_embed.append(entity_image_embeds.cpu())
                outputs_text_seq_tokens.append(entity_text_seq_tokens.cpu())
                outputs_image_patch_tokens.append(entity_image_patch_tokens.cpu())

        self.entity_text_embeds = torch.concat(outputs_text_embed, dim=0)
        self.entity_image_embeds = torch.concat(outputs_image_embed, dim=0)
        self.entity_text_seq_tokens = torch.concat(outputs_text_seq_tokens, dim=0)
        self.entity_image_patch_tokens = torch.concat(outputs_image_patch_tokens, dim=0)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch_raw = self.trainer.datamodule.test_raw[
                    batch_idx * self.args.data.eval_batch_size: (batch_idx + 1) * self.args.data.eval_batch_size]
        answer = batch.pop('answer')
        assert answer.tolist() == [self.trainer.datamodule.qid2id[_['answer']] for _ in
                                   batch_raw]
        batch_size = len(answer)
        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**batch)

        scores = []
        chunk_size = self.args.data.eval_chunk_size
        for idx in range(math.ceil(self.args.data.num_entity / chunk_size)):
            start_pos = idx * chunk_size
            end_pos = (idx + 1) * chunk_size

            chunk_entity_text_embeds = self.entity_text_embeds[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_image_embeds = self.entity_image_embeds[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_text_seq_tokens = self.entity_text_seq_tokens[start_pos:end_pos].to(mention_text_embeds.device)
            chunk_entity_image_patch_tokens = self.entity_image_patch_tokens[start_pos:end_pos].to(
                mention_text_embeds.device)

            chunk_score, _ = self.matcher(chunk_entity_text_embeds, chunk_entity_text_seq_tokens,
                                          mention_text_embeds, mention_text_seq_tokens,
                                          chunk_entity_image_embeds, chunk_entity_image_patch_tokens,
                                          mention_image_embeds, mention_image_patch_tokens)
            scores.append(chunk_score)

        scores = torch.concat(scores, dim=-1)
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()
        return dict(rank=tgt_rank, all_rank=rank.detach().cpu().numpy(), scores=scores.detach().cpu().numpy())

    def test_epoch_end(self, outputs):
        self.entity_text_embeds = None
        self.entity_image_embeds = None
        self.entity_text_seq_tokens = None
        self.entity_image_patch_tokens = None

        ranks = np.concatenate([_['rank'] for _ in outputs])
        # rerank
        if self.args.rerank:
            all_ranks = np.concatenate([_['all_rank'] for _ in outputs])
            all_score = np.concatenate([_['scores'] for _ in outputs])
            all_ranks = torch.tensor(all_ranks).to(torch.device('cuda:0'))
            all_score = torch.tensor(all_score).to(torch.device('cuda:0'))
            filtered_indices = filter_by_entropy(all_score, self.entropy_l_thresh, self.entropy_h_thresh)[
                0].tolist()
            self.fuzzy_count += len(filtered_indices)
            test_raw = self.trainer.datamodule.test_raw
            for mention_index in tqdm(filtered_indices):
                mention_name = test_raw[mention_index]['mentions']
                mention_text = test_raw[mention_index]['sentence']
                mention_pic_path = test_raw[mention_index]['imgPath']
                topk_cddt_indices = torch.topk(all_score[mention_index],
                                               self.topk).indices.tolist()
                candidates = []
                for entity_index in topk_cddt_indices:
                    entity_name = self.trainer.datamodule.raw_kb_entity[entity_index]['entity_name']
                    entity_desc = self.trainer.datamodule.raw_kb_entity[entity_index]['desc']
                    candidates.append({'index': entity_index, 'name': entity_name, 'text': entity_desc})
                try:
                    number = int(LLM_rerank(mention_name, mention_text, mention_pic_path, candidates,
                                            self.args.data.mention_img_folder, self.topk))
                    top1index = candidates[number]['index']
                    all_ranks = rerank_topk(all_ranks, mention_index, top1index, topk_cddt_indices)
                except:
                    pass
            answer = torch.tensor([_['answer'] for _ in self.trainer.test_dataloaders[0].dataset])
            ranks = all_ranks[torch.arange(len(answer)), answer].detach().cpu().numpy()

        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Test/hits20", hits20)
        self.log("Test/hits10", hits10)
        self.log("Test/hits5", hits5)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mr", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())
        self.log("Samples", len(ranks))
        self.log("Fuzzy Samples", self.fuzzy_count)
        self.log("Fuzzy Ratio", self.fuzzy_count / len(ranks))
        self.fuzzy_count = 0

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0001},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-4)
        return [optimizer]

    def compute_teacher_soft_labels(self, x_logits):
        return softmax(x_logits / self.temperature, dim=1)

    def compute_loss(self, x_logits, labels):
        one_hot_label = torch.nn.functional.one_hot(labels).float()
        return - torch.sum(one_hot_label * self.logsoftmax(x_logits.detach()), 1, keepdim=True)

    def compute_weighted_loss(self, x_logits, soft_labels_x, loss_t_x, labels):
        loss_s_x = self.compute_loss(x_logits, labels)
        focal_weight_x = loss_s_x / (loss_t_x + 1e-7)
        focal_weight_x = torch.max(focal_weight_x, torch.zeros(1).cuda())
        focal_weight_x = 1 - torch.exp(- focal_weight_x)
        loss_x = - torch.sum(soft_labels_x * self.logsoftmax(x_logits / self.temperature), 1, keepdim=True)
        loss_x = focal_weight_x * loss_x
        loss_x = (self.temperature ** 2) * torch.mean(loss_x)
        x_loss = self.loss_fct(x_logits / self.temperature, labels)
        return x_loss + self.alpha * loss_x
