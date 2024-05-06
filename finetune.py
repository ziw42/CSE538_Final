from transformers import AdamW
from utils.utils import loadModel, MRMDataset, int2str, createLoader
import torch
from tqdm import tqdm

def finetune(model_name, ds_name, **kwargs):
    ### Set hyperparameters
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    num_epochs = kwargs["num_epochs"]
    max_length = kwargs["max_length"]

    ### Load model
    # LLaMA2 and GPT2 as 
    if "llama2" in model_name:
        tokenizer, model = loadModel("meta-llama/Llama-2-7b-chat-hf")
    elif "opt" in model_name:
        tokenizer, model = loadModel("facebook/opt-1.3b")
    # OPT and BERT as baseline models
    elif "gpt2" in model_name:
        tokenizer, model = loadModel("openai-community/gpt2")
    elif "bert" in model_name:
        tokenizer, model = loadModel("google-bert/bert-base-uncased")
    else:
        print("Model not found.")
        raise ValueError

    ### Load data
    loader = createLoader(ds_name, tokenizer, batch_size, split=ds_name, max_length=256)

    ### Set optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print("######Start training######")

    ### Training loop
    for epoch in range(num_epochs):  
        for batch in tqdm(loader):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    ### Evaluate the model
    model.eval()
    predictions = []
    loader = createLoaderMRM(tokenizer, batch_size, split="test")
    for batch in loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
        predictions += torch.argmax(logits, dim=-1).tolist()

    ### Save the finetuned model
    model.save_pretrained('/home/jkl6486/fknews')
    print()