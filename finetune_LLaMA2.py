"""
CSE 538: Assignment 3
Team Spirit: Zian Wang, Yukun Yang
System: Ubuntu 20.04, Python 3.11.0, with Intel Xeon 4214, 128gb RAM, NVIDIA RTX A6000 * 4

This script finetunes the LLaMA-2 model.

"""
from transformers import AdamW
from utils.utils import loadModel, MRMDataset, int2str, createLoaderMRM
import torch
from tqdm import tqdm
from finetune import finetune

if __name__ == "__main__":
    ### Set hyperparameters
    batch_size = 2
    lr = 1e-5
    num_epochs = 1
    max_length = 256
    dataset_name = "mrm8488/fake-news"

    finetune("meta-llama/Llama-2-7b-chat-hf", 
             dataset_name, 
             batch_size=batch_size, 
             learning_rate=lr, 
             num_epochs=num_epochs, 
             max_length=max_length, 
             vocab=False, 
             train=True, 
             evaluation=True, 
             save_step=100)




