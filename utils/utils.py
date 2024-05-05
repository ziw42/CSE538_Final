import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def loadModel(model_name="meta-llama/Llama-2-7b-chat-hf"):
    ### Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    ### Load LLaMA2
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def printMetrics(predictions, labels, title):
    # Function to calculate metrics in this task
    # Calculate the metrics
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    precision_yes = precision_score(labels, predictions, pos_label=1)
    recall_yes = recall_score(labels, predictions, pos_label=1)
    f1_yes = f1_score(labels, predictions, pos_label=1)
    precision_no = precision_score(labels, predictions, pos_label=0)
    recall_no = recall_score(labels, predictions, pos_label=0)
    f1_no = f1_score(labels, predictions, pos_label=0)
    print(title)
    print("Overall: acc: " + str(accuracy) + ", f1: " + str(macro_f1))
    print("Yes: acc: " + str(precision_yes) + ", rec: " + str(recall_yes) + ", f1: " + str(f1_yes))
    print("No: acc: " + str(precision_no) + ", rec: " + str(recall_no) + ", f1: " + str(f1_no))

def int2str(i):
    if i == 1:
        return("True")
    else:
        return("False")

class MRMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = "Is the following statement true or false?: "+self.texts[idx]
        answer = prompt + self.labels[idx]
        text_encodings = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length)
        label_encodings = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length)

        item = {key: torch.tensor(val) for key, val in text_encodings.items()}
        item['labels'] = torch.tensor(label_encodings['input_ids']) 
        return item
    
def preMRM(tokenizer, batch_size, split="train", max_length=512):
    ### Load dataset
    dataset = load_dataset("mrm8488/fake-news", split=split)   

    # 编码数据集
    texts = dataset['text']  # dataset['text'] 应该是一个包含所有文本的列表
    labels = dataset['label']  # dataset['label'] 是对应的标签列表
    labels = [int2str(l) for l in labels]

    # 创建 CustomDataset 实例
    dataset = MRMDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=max_length)

    # 设置 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

    

