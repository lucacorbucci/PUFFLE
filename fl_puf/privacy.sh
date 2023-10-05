# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.20 --weight_decay_lambda 0.001 &
# wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # # Fixed Lambda 0.5
# # poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# # wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.20 --weight_decay_lambda 0.001 &
# wait;

# # Baseline no DPL
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000.0 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;


# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.1 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 1.0 --weight_decay_lambda 0.1 &
# wait;


# Experiments with 100 nodes balanced-unbalanced

# 7 unbalanced nodes - Baseline
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 &
# wait;

# 3 unbalanced nodes 
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 &
wait;

# 7 unbalanced nodes - Private
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 &
wait;

# 3 unbalanced nodes 
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 &
wait;

# _________________________________________________________________

# Vedere i precedenti e scegliere i parametri ANCORA DA ESEGUIRE

# # 7 unbalanced nodes - Baseline
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;


# # 7 unbalanced nodes - Baseline
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.7 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;

# # 3 unbalanced nodes 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 1.0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --partition_ratio 0.3 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;


# _________________________________________________________________

# Tunable Lambda starting from 0 every time
poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
wait;

# Tunable Lambda with bigger alpha
poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.5 --target 0.10 --weight_decay_lambda 0.001 &
wait;

# Tunable Lambda starting from 0 every time
poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
wait;


# _________________________________________________________________
# Test on the Original Celeba Dataset

# Original Celeba IID 
poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
wait;

# Original Celeba Non IID
poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
wait;

# Vedere i precedenti e scegliere i parametri ANCORA DA ESEGUIRE

# # Original Celeba IID with Tunable Lambda
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 100000000 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001  &
# wait;

# # Original Celeba Non IID with Tunable Lambda 
# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001  &
# wait;


# _________________________________________________________________
# Run for 200 epochs


# Baseline no DPL with Privacy
poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
wait;

# Fixed Lambda 0.5 with Privacy
poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
wait;

# Tunable Lambda with Privacy
poetry run python main.py --num_rounds 200 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.3 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 &
wait;

poetry run python main.py --num_rounds 5 --dataset celeba --epochs 1 --num_client_cpus 2 --num_client_gpus 2 --batch_size 512 --pool_size 4 --sampled_clients 1 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 5.0 --noise_multiplier 2.0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.001 