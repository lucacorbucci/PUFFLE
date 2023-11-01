# # Fixed value
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0 --momentum 0 --wandb True &
# wait;

# # Fixed value
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0.5 --momentum 0 --wandb True &
# wait;

# # avg
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode avg --momentum 0 --wandb True &
# wait;

# # disparity
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0 --wandb True &
# wait;


# # Fixed value with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0 --momentum 0.6 --wandb True &
# wait;

# # Fixed value with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0.5 --momentum 0.6 --wandb True &
# wait;

# avg with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode avg --momentum 0.6 --wandb True &
# wait;

# # disparity with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 3.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0.6 --wandb True &
# wait;





# # Fixed value
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0 --momentum 0 --wandb True &
# wait;

# # Fixed value
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0.5 --momentum 0 --wandb True &
# wait;

# # avg
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode avg --momentum 0 --wandb True &
# wait;

# # disparity
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0 --wandb True &
# wait;


# # Fixed value with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0 --momentum 0.6 --wandb True &
# wait;

# # Fixed value with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode fixed --starting_lambda_value 0.5 --momentum 0.6 --wandb True &
# wait;

# # avg with Momentum
# poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode avg --momentum 0.6 --wandb True &
# wait;

# disparity with Momentum
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0.3 --wandb True &
wait;


# disparity with Momentum
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.25 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0.6 --wandb True &
wait;

# disparity with Momentum
poetry run python main.py --num_rounds 50 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 0.2 --batch_size 512 --pool_size 100 --sampled_clients 0.45 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0 --wandb True &
wait;


poetry run python main.py --num_rounds 50 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1.0 --batch_size 512 --pool_size 10 --sampled_clients 0.45 --sampled_clients_test 0.35 --sampled_clients_validation 0.35 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 100000.0  --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41 --optimizer sgd --alpha_target_lambda 1.0 --target 0.1 --weight_decay_lambda 0.0001 --noise_multiplier 0 --training_nodes 0.4 --validation_nodes 0.3 --test_nodes 0.3 --node_shuffle_seed 99 --starting_lambda_mode disparity --momentum 0
