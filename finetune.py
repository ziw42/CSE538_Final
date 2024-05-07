"""
CSE 538: Assignment 3
Team Spirit: Zian Wang, Yukun Yang
System: Ubuntu 20.04, Python 3.11.0, with Intel Xeon 4214, 128gb RAM, NVIDIA RTX A6000 * 4

This script provides a wrapper function to finetune the model on the dataset.
model_name could be "llama2", "opt", "gpt2", "bert"
ds_name could be "fakenewsnet", "mrm8488", "liar"
The rest hyperparameters (batch size, number of epochs, learning rate, etc.) are passed in kwargs

"""

from transformers import AdamW
from utils.utils import loadModel, int2str, createLoader, logit2label2, logit2label6, printMetrics, loadModelFinetuned, label2int, logit2label62
import torch
from tqdm import tqdm


def finetune(model_name, ds_name, **kwargs):
    """
    [I. Syntax]
    [II. Semantics]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ### Use CUDA by default, else CPU

    ### Load model
    if kwargs["continued"]:
        # LLaMA2 and GPT2
        if "llama2" in model_name:
            tokenizer, model = loadModelFinetuned("meta-llama/Llama-2-7b-chat-hf", kwargs["model_path"])
            checkpoint_path = "/home/jkl6486/fknews/checkpoints/llama2"
        elif "opt" in model_name:
            tokenizer, model = loadModelFinetuned("facebook/opt-1.3b", kwargs["model_path"])
            checkpoint_path = "/home/jkl6486/fknews/checkpoints/opt"
        # OPT and BERT as baseline models
        elif "gpt2" in model_name:
            tokenizer, model = loadModelFinetuned("openai-community/gpt2", kwargs["model_path"])
            checkpoint_path = "/home/jkl6486/fknews/checkpoints/gpt"
        elif "bert" in model_name:
            tokenizer, model = loadModelFinetuned("google-bert/bert-base-uncased", kwargs["model_path"])
            checkpoint_path = "/home/jkl6486/fknews/checkpoints/bert"
        else:
            print("Model not found.")
            raise ValueError
    else:
        # LLaMA2 and GPT2
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
    loader = createLoader(ds_name, tokenizer, batch_size, split="train", max_length=max_length, model_name=model_name)

    ### Set optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print("###### Start training ######")

    if train:
        ### If we are finetuning BERT, we use MSE loss
        bert = False
        if "bert" in model_name:
            bert = True   ### We use a boolean here to make the judgement later quicker since it does not need to compare strings later
            loss_fn = torch.nn.MSELoss()
        ### Training loop
        for epoch in range(num_epochs):
            batch_count = 0
            for batch in tqdm(loader):
                batch_count += 1
                ### Move data to device
                input = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                if bert:
                    outputs = model(input_ids=input, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits.squeeze(), labels.squeeze()) ### If finetuning BERT, calculate MSELoss
                else:
                    outputs = model(input_ids=input, attention_mask=attention_mask, labels=labels) ### Input labels to calculate loss
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
                    try:
                        torch.save(checkpoint, checkpoint_path + "/checkpoint_" + str(epoch) + "_" + str(batch_count) + ".pth")
                    except:
                        pass

    if evaluation:
        ### Evaluate the model
        yes_id = tokenizer.encode("true", add_special_tokens=False)[0]
        no_id = tokenizer.encode("false", add_special_tokens=False)[0]

        model.eval()
        prediction_list = []
        labels_list = []
        loader = createLoader(ds_name, tokenizer, batch_size=1, split="test", max_length=max_length, model_name=model_name)
        for batch in loader:
            ### Move data to device
            input = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if "bert" in model_name:
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input, attention_mask=attention_mask)
                logits = outputs.logits
                for logit in logits:
                    if "liar" in ds_name:
                        prediction_list += [logit2label6(logit.item())]
                        labels_list.append(logit2label6(labels.item()))
                    else:
                        prediction_list += [logit2label2(logit.item())]
                        labels_list.append(labels.item())
            else:
                labels = batch["text_labels"]
                outputs = model.generate(input, max_length=input.shape[1] + 30, output_scores=True, return_dict_in_generate=True)
                try:
                    logits = outputs.scores[5]  ### The sixth token is the expected answer: " \n Answer: [true or false]"
                except:
                    logtis = outputs.scores[0] ### If the model gave very short answer, we use the first token
                    print("Very short answer here, use the first token instead.")
                prediction_list += [1 if logits[0][yes_id] > logits[0][no_id] else 0]
                if "liar" in ds_name:
                    labels_list.append(logit2label62(labels.item()))
                else:
                    labels_list.append(label2int(labels))
        print()
        printMetrics(prediction_list, labels_list, ds_name, title=f"Finetuned {model_name} on {ds_name} dataset")

    ### Save the finetuned model
    model.save_pretrained('/home/jkl6486/fknews/model/' + model_name)
    print()










