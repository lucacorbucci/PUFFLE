# Baseline
poetry run python ../../main.py --batch_size=138 --epochs=5 --lr=0.06562545490251828 --node_shuffle_seed=933317809 --optimizer=adamw --dataset celeba --num_rounds 25 --num_client_cpus 1 --num_client_gpus 0.2 --pool_size 150 --sampled_clients 0.20 --sampled_clients_test 1 --train_csv original_merged --debug True --base_path ../../../data --private True --delta 0.0001 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --sweep True --noise_multiplier 0 --clipping 100000 --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode no_tuning --percentage_unbalanced_nodes 0.5 --unbalanced_ratio 0.5 --dataset_path ../../../data/celeba

# Fixed 
poetry run python ../../main.py --batch_size=141 --epochs=5 --lr=0.09087111636953207 --node_shuffle_seed=26418572 --optimizer=adamw --starting_lambda_value=0.43735640745167687 --dataset celeba --num_rounds 25 --num_client_cpus 1 --num_client_gpus 0.2 --pool_size 150 --sampled_clients 0.20 --sampled_clients_test 1 --train_csv original_merged --debug True --base_path ../../../data --DPL True --private True --delta 0.0001 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --sweep True --noise_multiplier 0 --clipping 100000 --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode fixed --target 0.05 --percentage_unbalanced_nodes 0.5 --unbalanced_ratio 0.5 --dataset_path ../../../data/celeba

# Tunable 
poetry run python ../../main.py --alpha_target_lambda=0.39837983363978846 --batch_size=178 --epochs=5 --lr=0.09822031939202236 --momentum=0.07578082387773348 --node_shuffle_seed=991867416 --optimizer=adam --weight_decay_lambda=0.3418715042816064 --dataset celeba --num_rounds 25 --num_client_cpus 1 --num_client_gpus 0.2 --pool_size 150 --sampled_clients 0.20 --sampled_clients_test 1 --train_csv original_merged --debug True --base_path ../../../data --DPL True --private True --delta 0.0001 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --target 0.05 --sweep True --noise_multiplier 0 --clipping 100000 --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode disparity --update_lambda True --percentage_unbalanced_nodes 0.5 --unbalanced_ratio 0.5 --dataset_path ../../../data/celeba