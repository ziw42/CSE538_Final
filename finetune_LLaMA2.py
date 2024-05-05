from transformers import AdamW
from utils.utils import loadModel, MRMDataset, int2str, preMRM
import torch
from tqdm import tqdm

### Set hyperparameters
batch_size = 4
lr = 1e-5
num_epochs = 1
max_length = 256

### Load model
tokenizer, model = loadModel("meta-llama/Llama-2-7b-chat-hf")

### Load data
loader = preMRM(tokenizer, batch_size, split="train", max_length=256)

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
loader = preMRM(tokenizer, batch_size, split="test")
for batch in loader:
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits
    predictions += torch.argmax(logits, dim=-1).tolist()




### Save the finetuned model
model.save_pretrained('/home/jkl6486/fknews')
print()





