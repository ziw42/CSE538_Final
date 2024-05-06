import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def loadModel(model_name="meta-llama/Llama-2-7b-chat-hf"):
    ### Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    ### Load model
    ### If BERT, we use AutoModelForSequenceClassification
    if "bert" in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
    ### Else it is a CausalLM
    else:
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

def int2str(i, dataset_name):
    """
    Convert labels in each dataset to the uniform label
    params:
        i: int(str), label
        dataset_name: str, dataset name
    return:
        str, label
    """
    ### MRM fakenews dataset
    ### 0 = fake, 1 = true
    if "mrm8488" in dataset_name:
        if i == 1:
            return("true")
        else:
            return("fake")
    ### LIAR fakenews dataset
    ### 0 = fake, 1 = half-true, 2 = mostly-true, 3 = true, 4 = barely-true, 5 = pants-fire
    elif "liar" in dataset_name:
        if i == 0:
            return("fake")
        elif i == 1:
            return("half-true")
        elif i == 2:
            return("mostly-true")
        elif i == 3:
            return("true")
        elif i == 4:
            return("barely-true")
        else:
            return("pants-fire")
    ### FakeNewsNet dataset
    ### fake = fake, real = true
    elif "fakenewsnet" in dataset_name:
        if i == "fake":
            return("fake")
        else:
            return("true")


class FakeNewsDataset(Dataset):
    """
    Class for customized fakenews dataset container
    params:
        texts: list, list of texts
        labels: list, list of labels
        tokenizer: tokenizer, tokenizer
        max_length: int, max length for encoding
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ### Build the prompts
        prompt = "Is the following statement true or fake?: " + self.texts[idx]
        ### Because we use the causal language model, we need to append the label to the prompt as the labels for finetuning
        answer = prompt + "\nAnswer: " + self.labels[idx]
        ### Encode the prompt and label
        text_encodings = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length)
        label_encodings = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length+10) ### Save the space for label

        item = {key: torch.tensor(val) for key, val in text_encodings.items()}
        item['labels'] = torch.tensor(label_encodings['input_ids']) 
        return item

def truncateText(text, max_length, tokenizer):
    """
    Truncate the text to max_length
    params:
        text: str, text
        max_length: int, max length
        tokenizer: tokenizer, tokenizer
    return:
        str, truncated text
    """
    encoded_text = tokenizer.encode(text)
    ### Truncate the text if it is longer than max_length
    if len(encoded_text) > max_length:
        truncated = encoded_text[:max_length]
    else:
        truncated = encoded_text
    ### Decode the truncated text
    truncated_text = tokenizer.decode(truncated, clean_up_tokenization_spaces=True)

    return truncated_text

    
def createLoader(dataset_name, tokenizer, batch_size, split="train", max_length=512, shuffle=True):
    """
    Load dataset and create the DataLoader wrapper
    params:
        dataset_name: str, dataset name
        tokenizer: tokenizer, tokenizer
        batch_size: int, batch size
        split: str, split name
        max_length: int, max length
    return:
        DataLoader, DataLoader
    """
    ### Load dataset
    if "fakenewsnet" not in dataset_name:
        dataset = load_dataset(dataset_name, split=split)   
        ### Encode dataset
        if "mrm8488" in dataset_name:
            ### Extract each column
            texts = truncateText(dataset['text']) 
            labels = dataset['label']
            ### Convert 0, 1 label to "true", "fake"
            labels = [int2str(l) for l in labels]
        elif "liar" in dataset_name:
            texts = truncateText(dataset["statement"])
            labels = dataset["label"]
    else:
        texts = []
        labels = []
        with open("/home/jkl6486/fknews/data/FakeNewsNet_Data.jsonl", "r") as f:
            for l in f:
                data = json.loads(l)
                texts.append(truncateText(data["text"]))
                labels.append(data["label"])
                labels = [int2str(l, dataset_name) for l in labels]

    ### Create dataset object
    dataset = FakeNewsDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=max_length+30) ### Save the space for template in the prompt

    ### Create DataLoader object
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

    

