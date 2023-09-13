import os
import sys
import time
from typing import List
from baseline.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel
import pandas as pd

import fire
import torch
import pickle
import numpy as np
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from model import LLM4Rec
from data_utils import BipartiteGraphDataset, BipartiteGraphCollator, SequentialDataset, SequentialCollator

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import gc

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "",
        task_type: str = "",
        # lora hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
        lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
        # llm hyperparams
        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca"
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    prompter = Prompter(prompt_template_name)

    if task_type == 'general':
        dataset = BipartiteGraphDataset(data_path)
        user_embed, item_embed = (pickle.load(open('datasets/general/' + data_path + '/VanillaMF_user_embed.pkl', 'rb')),
                                  pickle.load(open('datasets/general/' + data_path + '/VanillaMF_item_embed.pkl', 'rb')))
        model = LLM4Rec(
            base_model=base_model,
            input_dim=64,
            output_dim=dataset.m_item,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=device_map,
            instruction_text=prompter.generate_prompt(task_type),
            user_embeds=user_embed,
            input_embeds=item_embed,
        )
    elif task_type == 'sequential':
        dataset = SequentialDataset(data_path, 50)
        input_embed = pickle.load(open('datasets/sequential/' + data_path + '/SASRec_item_embed.pkl', 'rb'))
        model = LLM4Rec(
            base_model=base_model,
            input_dim=64,
            output_dim=dataset.m_item,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            device_map=device_map,
            instruction_text=prompter.generate_prompt(task_type),
            user_embeds=None,
            input_embeds=input_embed,
        )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    topk = [5, 10, 20]
    testData = dataset.testData
    users = np.arange(dataset.n_user)

    results = {'Precision': np.zeros(len(topk)),
               'Recall': np.zeros(len(topk)),
               'MRR': np.zeros(len(topk)),
               'MAP': np.zeros(len(topk)),
               'NDCG': np.zeros(len(topk))}
    for u in users:
        all_pos = dataset.allPos[u]
        groundTruth = [testData[u]]

        inputs = torch.LongTensor(testData[u]).cuda().unsqueeze(0)

        ratings = model.predict(inputs)
        exclude_index = []
        exclude_items = []
        for range_i, its in enumerate(all_pos):
            exclude_index.extend([range_i] * len(its))
            exclude_items.extend(its)
        ratings[exclude_index, exclude_items] = -(1 << 10)
        _, ratings_K = torch.topk(ratings, k=topk[-1])
        ratings_K = ratings_K.cpu().numpy()

        r = getLabel(groundTruth, ratings_K)
        for j, k in enumerate(topk):
            pre, rec = RecallPrecision_atK(groundTruth, r, k)
            mrr = MRR_atK(groundTruth, r, k)
            map = MAP_atK(groundTruth, r, k)
            ndcg = NDCG_atK(groundTruth, r, k)
            results['Precision'][j] += pre
            results['Recall'][j] += rec
            results['MRR'][j] += mrr
            results['MAP'][j] += map
            results['NDCG'][j] += ndcg

    for key in results.keys():
        results[key] /= float(len(users))
    print(f'Evaluation for User: \n')
    for j, k in enumerate(topk):
        print(f'Precision@{k}: {results["Precision"][j]} \n '
              f'Recall@{k}: {results["Recall"][j]} \n '
              f'MRR@{k}: {results["MRR"][j]} \n '
              f'MAP@{k}: {results["MAP"][j]} \n '
              f'NDCG@{k}: {results["NDCG"][j]} \n')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

