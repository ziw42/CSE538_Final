#===============================================================================
# usecommand: nohup bash shell/finetune_OPT.sh > shell/0506_finetune_OPT.log 2>&1 &
#===============================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_OPT.py

