import torch
import base64
from openai import OpenAI
from omegaconf import OmegaConf
from torch.nn.functional import softmax


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    return args


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def filter_by_entropy(scores: torch.tensor, lthreshold: float, hthreshold):
    entropy = -torch.sum(softmax(scores, dim=1) * torch.log(softmax(scores, dim=1) + 1e-10), dim=1)
    indices = torch.where((entropy > lthreshold * torch.max(entropy)) & (entropy < hthreshold * torch.max(entropy)))
    return indices


def LLM_rerank(mention_name, mention_text, mention_pic_path, candidate_dicts, mention_img_folder, topk):
    if mention_pic_path != '' and mention_pic_path is not None:
        base64_image = encode_image(mention_img_folder + "/" + mention_pic_path)

    client = OpenAI(  # todo you need to modify this
        api_key="sk-",
        base_url=" "
    )

    PROMPT_head = f"""
    ###Mention
    Name: {mention_name}
    Context: {mention_text}
    """
    PROMPT_body = """
    ###Entity table
    """
    PROMPT_tail = """
    Just give the serial number and do not give me any other information.
    The most matched serial number is:
    """

    for i in range(topk):
        PROMPT_body += f"""{i}. 
        Name:{candidate_dicts[i]['name']}
        Description:{candidate_dicts[i]['text']}\n\t"""

    PROMPT = PROMPT_head + PROMPT_body + PROMPT_tail

    if mention_pic_path != '' and mention_pic_path is not None:
        message = [
            {"role": "system",
             "content": "You are an expert in selecting the best-matched entity to match the given mention, note that the picture belongs to the mention. Compare names in mentions with each entity in the entity table and prioritize exact matches or the closest ones firstly, then check if the description of the entity is more closely related to the mention's context."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                ],
            }
        ]
    else:
        message = [
            {"role": "system",
             "content": "You are an expert in selecting the best-matched entity to match the given mention. Compare names in mentions with each entity in the entity table and prioritize exact matches or the closest ones firstly, then check if the description of the entity is more closely related to the mention's context."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                ],
            }
        ]
    response = client.chat.completions.create(
        model="qwen2.5-vl-7b-instruct",
        messages=message,
    )
    entity = response.choices[0].message.content
    return entity


# 将前k个candidates重新排序，目的是将大模型给出的top1放到最前
def rerank_topk(rank: torch.Tensor, mention_index, top1index, topk_cddt_indices):
    if rank[mention_index, top1index].item() == 1:
        return rank
    # 如果大模型给出的top1不在最前，那么将其排名置于1，在它前面的所有candidates排名+1
    else:
        current_rank = rank[mention_index, top1index].item()
        for _ in topk_cddt_indices:
            a = rank[mention_index, _].item()
            if a < current_rank:
                rank[mention_index, _] = a + 1
        rank[mention_index, top1index] = 1
        return rank
