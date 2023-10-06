## Source code for the experiments in the NeurIPS 2023 accepted paper Knowledge Distillation Performs Partial Variance Reduction


The code contains the scripts for training models using regular SGD and knowledge distillation, 
as well as utilities for tracking different statistics to measure the discrepancy between standard KD and 
the distillation gradient introduced in the paper. The arXiv version of the paper can be found at the following
[link](https://arxiv.org/abs/2305.17581).

The code is organized as follows:
* ``utils/dataset_utils.py`` contains functions for creating the datasets (MNIST, CIFAR-10), as well as 
for pre-extracting ResNet50 features
* ``utils/checkpoint_utils.py`` functions for loading and saving checkpoints during training
* ``utils/kd_utils.py`` specific functions for training using knowledge distillation (the standard version and 
the one using the distillation gradient introduced in this work), as well as utilities for tracking various 
statistics related to KD training
* ``utils/train_utils.py`` higher level functions for training and testing the models
* ``train_convex_KD_mnist.py`` the script for training linear MNIST models using various forms of KD
* ``train_modified_kd.py`` the script for training linear CIFAR-10 features using various forms of KD
* ``train_nn_KD_mnist.py`` the script for training a simple NN on MNIST, using various forms of KD

To run and test the code, we provide a few exemplary scripts, which can be run, for example, via the command:
 
 ``bash run_kd_mnist_linear_release.sh ${gpu} ${teacher_path}``