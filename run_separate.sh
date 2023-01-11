#!/bin/bash

# This script runs the separate script for each of the 3 datasets

for dataset in EMNIST
do
python main_ete.py \
--dataset $dataset \
--choose_func save_feat \
--device_file p_NDI_soft_20221102 \
--device_test_num 1 \
--lr 0.003 \
--epoch 200 \
--batch 4

done
