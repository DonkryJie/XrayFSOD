#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/hl/Student/DYJ/one
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
for split in ${SPLIT[*]} 
do
  configfile=configs/fewshot/base/e2e_voc_split${split}_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  # rm last_checkpoint
  # python VOC_split${split}.py
done
