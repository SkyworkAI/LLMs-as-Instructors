# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import disable_caching
disable_caching()
from datasets import load_dataset
from torch.utils.data import Subset
import re
import os


class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    def get_prompt(self, sample):
        return

    def get_chosen(self, sample):
        return

    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank):
        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        self.raw_datasets = load_dataset("Dahoas/rm-static")

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']



class BelleOpenSoucreDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, data_file, eval_data_file=None):

        super().__init__(output_path, seed, local_rank)
        self.dataset_name = "BelleOpenSoucre"
        self.dataset_name_clean = "BelleOpenSoucre"
        dataset_cache_dir = "output/data_files"
        print("data_file = ", data_file)
        self.raw_datasets = load_dataset("json", data_files=data_file, cache_dir=dataset_cache_dir)
        self.raw_datasets.cleanup_cache_files()

        if eval_data_file!=None and os.path.exists(eval_data_file):
            print("eval_data_file = ", eval_data_file)
            self.dev_raw_datasets = load_dataset("json", data_files=eval_data_file, cache_dir=dataset_cache_dir)
            self.dev_raw_datasets.cleanup_cache_files()
            self.train_data = self.raw_datasets["train"]
            self.eval_data = self.dev_raw_datasets["train"]
        else:
            train_val = self.raw_datasets["train"].train_test_split(
                test_size=1000, shuffle=True, seed=42
            )
            self.train_data = train_val["train"]
            self.eval_data = train_val["test"]

        print("train_data: ", self.train_data)
        print("eval_data: ", self.eval_data)


    def get_train_data(self):
        return self.train_data

    def get_eval_data(self):
        return self.eval_data

    def get_conversations(self, sample):
        return sample
