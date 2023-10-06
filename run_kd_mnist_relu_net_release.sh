#!/bin/bash

gpu=$1
lr=0.01 
seed=42
teacher=$2

declare -a lbdas=(0.7)

echo train SGD model
CUDA_VISIBLE_DEVICES=${gpu} python train_nn_KD_mnist.py \
	--full --use_relu \
	--lr ${lr} --momentum 0. --weight_decay 0.0001 --batch_size 10 --epochs 50 \
	--seed ${seed} 


echo train KD model

for ((i=0;i<${#lbdas[@]};++i));
	do
	      echo CUDA_VISIBLE_DEVICES=${gpu} python train_nn_KD_mnist.py \
		--full --use_relu \
		--distill --distill_type standard --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0.0001 --batch_size 10 --epochs 50 \
		--lbda ${lbdas[i]} \
		--seed ${seed} \
		--measure_grads 

		CUDA_VISIBLE_DEVICES=${gpu} python train_nn_KD_mnist.py \
		--full --use_relu \
		--distill --distill_type standard --teacher_path ${teacher} \
		--lr ${lr} --momentum 0. --weight_decay 0.0001 --batch_size 10 --epochs 50 \
		--lbda ${lbdas[i]} \
		--seed ${seed} \
		--measure_grads
	done
