# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.20 --weight_decay_lambda 0.001 &
# wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # # Fixed Lambda 0.5
# # poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# # wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.20 --weight_decay_lambda 0.001 &
# wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;


# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.1 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 1.0 --weight_decay_lambda 0.1 &
# wait;


# Experiments with 100 nodes balanced-unbalanced

# 7 unbalanced nodes - Baseline
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 &
# wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 &
# wait;

# # 7 unbalanced nodes - Private
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 &
# wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 &
# wait;

# # _________________________________________________________________

# # Vedere i precedenti e scegliere i parametri ANCORA DA ESEGUIRE

# # # 7 unbalanced nodes - Baseline
# # poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# # wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;


# # 7 unbalanced nodes - Baseline
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;


# # _________________________________________________________________

# # Tunable Lambda starting from 0 every time
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda with bigger alpha
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.5 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda starting from 0 every time
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;


# # _________________________________________________________________
# # Test on the Original Celeba Dataset

# # Original Celeba IID 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Original Celeba Non IID
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Vedere i precedenti e scegliere i parametri ANCORA DA ESEGUIRE

# # # Original Celeba IID with Tunable Lambda
# # poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 100000000 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001  &
# # wait;

# # # Original Celeba Non IID with Tunable Lambda 
# # poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001  &
# # wait;


# # _________________________________________________________________
# # Run for 200 epochs


# # Baseline no DPL with Privacy
# poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5 with Privacy
# poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda with Privacy
# poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &
# wait;


# poetry run python main.py --num_rounds 3 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 10 --sampled_clients 0.3 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --validation_ratio 0.2 --wandb True --sweep True



# Experiment to compare the difference between the disparity computed 
# aggregating the disparity values and the disparity computed using 
# the statistics.

# Baseline without privacy and without DPL
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd &
# wait;

# Baseline with DPL
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &
# wait;

# Baseline with DPL partendo da 0
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.00001 &
# wait;


# test finali
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 2.0 --target 0.1 --weight_decay_lambda 0.00001 &
# wait;

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 &
# wait;




# Baseline without DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd  &
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd &
# wait;
# DPL with target 0.1
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &
# wait;

# DPL with target 0.1 partendo da 0
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 2.0 --target 0.1 --weight_decay_lambda 0.001 --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &
# wait;
# poetry run python main.py --num_rounds 100 --optimizer sgd --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &

## Still need to compute the noise based on the epsilon that we want to use.

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.5 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.00001 



# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 5 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10
# wait;

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 128 --pool_size 10 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10  &
# wait;


# Train, test, validation
poetry run python main.py --num_rounds 4 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --wandb True &
wait;

poetry run python main.py --num_rounds 4 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --wandb True &
wait;

# Train, test, validation
poetry run python main.py --num_rounds 4 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 10 --wandb True &
wait;


# # Train and Test
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.15 --sampled_clients_test 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.7 --test_nodes 0.3 --node_shuffle_seed 99 --wandb True &
# wait;

# # Train and Test
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 100 --sampled_clients 0.15 --sampled_clients_test 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.7 --test_nodes 0.3 --node_shuffle_seed 10 --wandb True &
# wait;



poetry run python main.py --num_rounds 4 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 0.5 --batch_size 512 --pool_size 10 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.2 --weight_decay_lambda 0.00001 --epsilon 10 --training_nodes 0.41 --validation_nodes 0.26 --test_nodes 0.33 --node_shuffle_seed 99 



poetry run python main.py --num_rounds 50 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 150 --sampled_clients 0.15 --sampled_clients_test 0.20 --sampled_clients_validation 0.34 --lr 0.1 --train_csv unfair_train  --debug True --base_path ../data --DPL True  --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.47 --validation_nodes 0.20 --test_nodes 0.335 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0