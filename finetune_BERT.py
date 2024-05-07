"""
CSE 538: Assignment 3
Team Spirit: Zian Wang, Yukun Yang
System: Ubuntu 20.04, Python 3.11.0, with Intel Xeon 4214, 128gb RAM, NVIDIA RTX A6000 * 4

This script finetunes the BERT model.

"""

from finetune import finetune
if __name__ == "__main__":
    ### Set hyperparameters
    batch_size = 4
    lr = 1e-5
    num_epochs = 2
    max_length = 256
    
    ### Finetune BERT on MRMFakeNews dataset
    finetune("bert", 
             "mrm8488/fake-news", 
             batch_size=batch_size, 
             learning_rate=lr, 
             num_epochs=num_epochs, 
             max_length=max_length, 
             vocab=False, 
             train=True, 
             evaluation=True, 
             save_step=3000,
             continued=False,
             model_path="/home/jkl6486/fknews/model/bert/")

    ### Keep finetuning BERT on LIAR dataset
    finetune("bert", 
             "liar", 
             batch_size=batch_size, 
             learning_rate=lr, 
             num_epochs=num_epochs, 
             max_length=max_length, 
             vocab=False, 
             train=True, 
             evaluation=True, 
             save_step=3000,
             continued=True,
             model_path="/home/jkl6486/fknews/model/bert/")

    ### Evaluate model on FakeNewsNet dataset
    finetune("bert", 
             "fakenewsnet", 
             batch_size=batch_size, 
             learning_rate=lr, 
             num_epochs=num_epochs, 
             max_length=max_length, 
             vocab=False, 
             train=False, 
             evaluation=True, 
             save_step=3000,
             continued=True,
             model_path="/home/jkl6486/fknews/model/bert/")




