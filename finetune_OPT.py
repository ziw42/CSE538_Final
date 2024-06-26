"""
CSE 538: Assignment 3
Team Spirit: Zian Wang, Yukun Yang
System: Ubuntu 20.04, Python 3.11.0, with Intel Xeon 4214, 128gb RAM, NVIDIA RTX A6000 * 4

This script finetunes the OPT model.

"""

from finetune import finetune
if __name__ == "__main__":
    ### Set hyperparameters
    batch_size = 4
    lr = 1e-5
    num_epochs = 1
    max_length = 256
    
    ### Finetune OPT on MRMFakeNews dataset
    finetune("opt", 
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
             model_path="/home/jkl6486/fknews/model/facebook/opt/")

    ### Keep finetuning OPT on LIAR dataset
    finetune("opt", 
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
             model_path="/home/jkl6486/fknews/model/opt/")

    ### Evaluate model on FakeNewsNet dataset
    finetune("opt", 
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
             model_path="/home/jkl6486/fknews/model/opt/")




