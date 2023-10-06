#!/bin/bash

gpu=$1
lr=0.05 
seed=42
teacher=$2

declare -a lbdas=(0.7)


# train the SGD model
CUDA_VISIBLE_DEVICES=${gpu} python train_convex_KD_mnist.py \
	--full \
	--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
	--seed ${seed}


# train the KD models

for ((i=0;i<${#lbdas[@]};++i));
	do
	      echo CUDA_VISIBLE_DEVICES=${gpu} python train_convex_KD_mnist.py \
		--full \
		--distill --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
		--lbda ${lbdas[i]} \
		--seed ${seed}

		CUDA_VISIBLE_DEVICES=${gpu} python train_convex_KD_mnist.py \
		--full \
		--distill --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0 --batch_size 10 --epochs 100 \
		--lbda ${lbdas[i]} \
		--seed ${seed}
	done
