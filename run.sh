#!/bin/bash

#SBATCH --job-name=cifar10_resnetv2
#SBATCH --output="./logs/resnetv2-res"
#SBATCH --nodelist=nv174
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "Run started at:- "
date

# ex) srun python -m mnist_resnet50.train

#srun python -m cifar10_resnetv2.train -model resnet110
#srun python -m cifar10_resnetv2.test -model resnet110
#srun python -m cifar10_resnetv2.train -model resnet164
#srun python -m cifar10_resnetv2.test -model resnet164
#srun python -m cifar10_resnetv2.train -model v2resnet110
srun python -m cifar10_resnetv2.test -model v2resnet110
#srun python -m cifar10_resnetv2.train -model v2resnet164
#srun python -m cifar10_resnetv2.test -model v2resnet164

#cnt=0
#while [ 1 == 1 ]
#do
#  if [ $cnt -eq 5 ]; then
#    break
#  fi
#  echo "Start loop after 5sec"
#  sleep 5
##  srun python -m cifar10_resnetv2.train -model resnet110
#  srun python -m cifar10_resnetv2.test -model resnet110
##  srun python -m cifar10_resnetv2.train -model resnet164
#  srun python -m cifar10_resnetv2.test -model resnet164
##  srun python -m cifar10_resnetv2.train -model v2resnet110
#  srun python -m cifar10_resnetv2.test -model v2resnet110
##  srun python -m cifar10_resnetv2.train -model v2resnet164
##  srun python -m cifar10_resnetv2.test -model v2resnet164
#  let cnt++
#  echo "Sleep for 10sec"
#  sleep 5
#done
