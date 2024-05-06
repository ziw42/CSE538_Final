from transformers import AdamW
from utils.utils import loadModel, MRMDataset, int2str, createLoader
import torch
from tqdm import tqdm


def finetune(model_name, ds_name, **kwargs):
    """
    Train the model
    params:
        model: model, model
        data_loader: DataLoader, DataLoader
        kwargs: dict, hyperparameters
    """
    ### Set hyperparameters
    batch_size = kwargs["batch_size"]
    lr = kwargs["learning_rate"]
    num_epochs = kwargs["num_epochs"]
    max_length = kwargs["max_length"]
    vocab = kwargs["vocab"] ### Print the loss for each batch
    train = kwargs["train"] ### Finetuning the model
    evaluation = kwargs["evaluation"] ### Evaluate the model
    save_step = kwargs["save_step"] ### Number of steps to save the checkpoint

    ### Load model
    # LLaMA2 and GPT2 as 
    if "llama2" in model_name:
        tokenizer, model = loadModel("meta-llama/Llama-2-7b-chat-hf")
        checkpoint_path = "/home/jkl6486/fknews/checkpoints/llama2"
    elif "opt" in model_name:
        tokenizer, model = loadModel("facebook/opt-1.3b")
        checkpoint_path = "/home/jkl6486/fknews/checkpoints/opt"
    # OPT and BERT as baseline models
    elif "gpt2" in model_name:
        tokenizer, model = loadModel("openai-community/gpt2")
        checkpoint_path = "/home/jkl6486/fknews/checkpoints/gpt"
    elif "bert" in model_name:
        tokenizer, model = loadModel("google-bert/bert-base-uncased")
        checkpoint_path = "/home/jkl6486/fknews/checkpoints/bert"
    else:
        print("Model not found.")
        raise ValueError

    ### Load data
    loader = createLoader(ds_name, tokenizer, batch_size, split="train", max_length=max_length)

    ### Set optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print("###### Start training ######")

    if train:
        ### Training loop
        for epoch in range(num_epochs):
            batch_count = 0
            for batch in tqdm(loader):
                batch_count += 1
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if vocab:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
                ### Save the model if reach the checkpoint
                if (batch_count + 1) % save_step == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch': batch_count,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }
                    torch.save(checkpoint, checkpoint_path + "/checkpoint_" + str(epoch) + "_" + str(batch_count) + ".pth")

    if evaluation:
        ### Evaluate the model
        model.eval()
        predictions = []
        loader = createLoader(ds_name, tokenizer, batch_size, split="test", max_length=max_length)
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits
            predictions += torch.argmax(logits, dim=-1).tolist()

    ### Save the finetuned model
    model.save_pretrained('/home/jkl6486/fknews/model/' + model_name)
    print()










