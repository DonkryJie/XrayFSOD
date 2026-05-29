#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/hl/students/DYJ/one
export CUDA_VISIBLE_DEVICES=1
export NGPUS=1
SPLIT=(1)
for split in ${SPLIT[*]} 
do
  configfile=configs/fewshot/base/e2e_voc_split${split}_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/demo.py --config-file ${configfile}
done