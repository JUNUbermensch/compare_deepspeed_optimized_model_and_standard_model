# -*- coding: utf-8 -*-
'''
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29502
'''

import deepspeed
import torch.optim as optim
import os
import torch
import time
import deepspeed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, load_metric
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AdamW
#from graph import Graph
from dataclasses import dataclass, field
#from utils import load_json
from glob import glob
#from generation_config import GenerationParams
#from bleu import Bleu
from typing import List
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def tokenize_function(examples):
    
    tokenized_inputs = tokenizer(
        [text for text in examples["context"]],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokenized_labels = tokenizer(
        examples["summary"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels["input_ids"],
    }

# model_name_or_path = "/data/suhyub/summarization/resources/t5_mss_small_torch"

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path
)

model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path
)

generation_config = GenerationConfig.from_pretrained(
        model_name_or_path
)

# file_path = '/data/suhyub/small_pretraining_t5_finetune_data_kbs_dialog_smr_v3_all_all (1) - 복사본.tsv'
learning_rate = 0.001
epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(file_path, sep="\t")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

with open("/data/suhyub/summarization/train_dataset.tsv", "w") as f:
    train_df.to_csv(f, sep='\t', index=False)

with open("/data/suhyub/summarization/test_dataset.tsv", "w") as f:
    test_df.to_csv(f, sep='\t', index=False)
    
batch_size = 8
train_dataset = Dataset.from_dict(train_df)
test_dataset = Dataset.from_dict(test_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

rouge = load_metric("rouge")

deepspeed_config = {
            "fp8": {"enabled": True},
            "train_batch_size": 128,
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 64,
            "zero_optimization": {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8
            },
            "zero_allow_untested_optimizer": True,
}

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='/data/suhyub/summarization/model_checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.F1_score_max = -float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, F1_score, model):
        score = F1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(F1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(F1_score, model)
            self.counter = 0
        
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.F1_score_max = -float('inf')
        print('Eearly stopping parameters are reset')
            
    def save_checkpoint(self, F1_score, model):
        '''Saves model when the F1 score increase.'''
        if self.verbose:
            self.trace_func(f'F1 score increased ({self.F1_score_max:.6f} --> {F1_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.F1_score_max = F1_score
        
early_stopping = EarlyStopping()

@torch.no_grad()
def predict(batch_loader, model, tokenizer, generation_config, device, verbose=False) :
    model.eval()
    model.to(device)

    predictions, references = [], []
    
    for batch in tqdm(batch_loader):
        label = batch.pop("labels", None)
        
        generation_config = {
          "max_length": 512,
          "temperature": 0.0,
          "top_k": 1,
          "top_p": 1,
          "num_beams": 1,
          "repetition_penalty" : 1.2 ,
          "early_stopping": False
        }

        # generation_config = {
        #   "max_length": 128,
        #   "temperature": 1.0,
        #   "top_k": 4,
        #   "top_p": 4,
        #   "num_beams": 4,
        #   "repetition_penalty" : 1.2 ,
        #   "early_stopping": True
        # }

        # generation_config = {
        #   "max_length": 128,
        #   "temperature": 0.0,
        #   "top_k": 1,
        #   "top_p": 1,
        #   "num_beams": 1,
        #   "repetition_penalty" : 1.2 ,
        #   "early_stopping": True
        # }
        ground_truths = tokenizer.batch_decode(batch['labels'].to(device), skip_special_tokens=True)
        references.extend(ground_truths)
        outputs = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            **generation_config
        )
        new_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(new_predictions)
        references = test_df['summary'].tolist()
        if verbose:
            print(f"Predictions: {new_predictions}")
            
    return predictions, references

def calculate_f1_score(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    rouge_l_f1 = results['rougeL'].mid.fmeasure
    return rouge_l_f1

def train_and_evaluate(model, dataloader, is_deepspeed=False):
    if is_deepspeed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        deepspeed.init_distributed(dist_backend="nccl", rank=local_rank, distributed_port=29502)
        d_model, _, _, _ = deepspeed.initialize(
            model=model, 
            optimizer=AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2), 
            config=deepspeed_config
        )
        d_optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(d_optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    max_f1 = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if is_deepspeed:
                outputs = d_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            if is_deepspeed:
                d_optimizer.step()
                d_optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if is_deepspeed:
                    outputs = d_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        num_beams=4,
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        num_beams=4,
                    )

                predictions.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                references.extend(batch["labels"].tolist())

        f1 = calculate_f1_score(predictions, references)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        max_f1 = max(max_f1, f1)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"F1 score: {f1:.5f}")
        print(f"max f1 score: {max_f1:.5f}")
        print(f"Processing time: {epoch_time:.2f} seconds")
        val_f1_score = calculate_f1_score(predictions, references)
        scheduler.step(val_f1_score)
        early_stopping(val_f1_score, model)
        if early_stopping.early_stop:
            model.load_state_dict(torch.load('/data/suhyub/summarization/model_checkpoint.pt'))
            print("Early stopping triggered")
            return max_f1
           
def main():
    print("Training standard T5 \n")
    start_time = time.time()
    max_f1_score_of_standard_t5 = train_and_evaluate(model.to(device), train_dataloader)
    end_time = time.time()
    standard_t5_time = end_time - start_time
    early_stopping.reset()
    print("\nTraining DeepSpeed-optimized T5:")
    start_time = time.time()
    max_f1_score_of_DeepSpeed_optimized_t5 = train_and_evaluate(model.to(device), train_dataloader, is_deepspeed=True)
    end_time = time.time()
    DeepSpeed_optimized_T5_time = end_time - start_time
    print(f'standard t5 time: {standard_t5_time} \nDeepSpeed optimized T5 time: {DeepSpeed_optimized_T5_time}')
    print(f'standard t5 f1 score: {max_f1_score_of_standard_t5} \nDeepSpeed optimized T5 f1 score: {max_f1_score_of_DeepSpeed_optimized_t5}')
    
if __name__ == "__main__":
    main()