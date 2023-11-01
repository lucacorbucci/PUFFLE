# # DLP with trainable Lambda with alpha 0.2 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.2 --target 0.10 &
# wait;
# # DLP with trainable Lambda with alpha 0.3 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.3 --target 0.10 &
# wait;
# # DLP with trainable Lambda with alpha 0.4 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.4 --target 0.10 &
# wait;


# # DLP with trainable Lambda with alpha 0.2 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.2 --target 0.10 &
# wait;
# # DLP with trainable Lambda with alpha 0.3 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.3 --target 0.10 &
# wait;
# # DLP with trainable Lambda with alpha 0.1 and target 0.05
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.1 --target 0.05 &
# wait;
# # DLP with trainable Lambda with alpha 0.2 and target 0.05
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.2 --target 0.05 &
# wait;
# # DLP with trainable Lambda with alpha 0.3 and target 0.05
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.3 --target 0.05 &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True &
# wait; 


## .-.

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.1 --target 0.1 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.2 --target 0.1 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.3 --target 0.1 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.1 --target 0.2 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.2 --target 0.2 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.3 --target 0.2 &
# wait;


# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.1 --target 0.15 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.2 --target 0.15 &
# wait; 

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True --alpha_target_lambda 0.3 --target 0.15 &
# wait;




# # Experiment with Trainable Lambda

# # DLP with trainable Lambda with alpha 0.4 and target 0.10
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.4 --target 0.10 &
# wait;

# # DLP with fixed Lambda 0.2
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.3
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.4
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.5
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.6
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.6 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.6
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.7 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;


# # Baseline 
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # I prossimi tre già eseguiti
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.01 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.1 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.01 &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.01 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.1 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.01 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.1 &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.1 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.01 --cross_silo True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0 --cross_silo True &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.01 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.1 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.01 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.1 --weight_decay_lambda True &
# wait;

# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.01 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.1 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.01 --weight_decay_lambda True &
# wait;
# poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.5 --target 0.10 --weight_decay_lambda 0.1 --weight_decay_lambda True &
# wait;


# # DLP with fixed Lambda 0.4
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.45
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.45 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # DLP with fixed Lambda 0.5
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;



# questa sotto ancora 

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.0001 &
# wait;

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 &
# wait;

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.0001 &
# wait;

# poetry run python main.py --num_rounds 100 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 &
# wait;




# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 &
# wait;



# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --cross_silo True &
# wait;


# Baseline no DPL
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.4
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.45
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.45 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Fixed Lambda 0.5
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;

# # Tunable Lambda 
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 1.0 --target 0.10 --weight_decay_lambda 0.001 &
# wait;

# test unbalanced and balanced with 50% unbalanced



# # Baseline 0.1
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug False --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.1 &
# wait;

# Baseline 0.3
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.3 &
# wait;

# # Baseline 0.5
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.5 &
# wait;

# # Baseline 0.7
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.7 &
# wait;


# CON DPL

# # Baseline 0.1
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug False --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.1 --alpha_target_lambda 1.0 --target 0.02 --weight_decay_lambda 0.001 &
# wait;

# Baseline 0.3
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.3 --alpha_target_lambda 1.0 --target 0.02 --weight_decay_lambda 0.001 &
# wait;

# # Baseline 0.5
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.5 --alpha_target_lambda 1.0 --target 0.02 --weight_decay_lambda 0.001 &
# wait;

# # Baseline 0.7
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.5 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.7 --alpha_target_lambda 1.0 --target 0.05 --weight_decay_lambda 0.001 &
# wait;



# Baseline 0.1
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug False --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.1 &
wait;

#Baseline 0.3
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.3 &
wait;

# Baseline 0.5
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.5 &
wait;
# # Baseline 0.7
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --no_sort_clients --percentage_unbalanced_nodes 0.7 &
# wait;


