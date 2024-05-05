#===============================================================================
# usecommand  nohup bash shell/finetune_LLaMA2.sh > shell/0505_finetune_LLaMA2.log 2>&1 &
#===============================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_LLaMA2.py

