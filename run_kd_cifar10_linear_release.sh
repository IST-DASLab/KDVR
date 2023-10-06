#!/bin/bash

gpu=$1
lr=0.05
seed=42
teacher=$2

declare -a lbdas=(0.7)

echo train the SGD model

CUDA_VISIBLE_DEVICES=${gpu} python train_modified_kd.py \
	--data cifar10 --data_path /home/Datasets/cifar10 \
	--full \
	--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
	--seed ${seed}



echo train the KD models

for ((i=0;i<${#lbdas[@]};++i)); 
	do
		echo CUDA_VISIBLE_DEVICES=${gpu} python train_modified_kd.py \
		--data cifar10 --data_path /home/Datasets/cifar10 \
		--full \
		--distill --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
		--lbda ${lbdas[i]} \
		--seed ${seed}

		CUDA_VISIBLE_DEVICES=${gpu} python train_modified_kd.py \
		--data cifar10 --data_path /home/Datasets/cifar10 \
		--full \
		--distill --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
		--lbda ${lbdas[i]} \
		--seed ${seed}
	done
