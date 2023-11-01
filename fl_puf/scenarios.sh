# DPL 
poetry run python main.py --alpha_target_lambda=1.3845378853715764 --batch_size=130 --epochs=3 --lr=0.07883081879897207 --momentum=0.3242433233377496 --node_shuffle_seed=838706845 --optimizer=adam --weight_decay_lambda=0.717999227519428 --dataset celeba --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.2 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1.0 --train_csv original_merged --debug True --base_path ../data --DPL True --private True --delta 0.0001 --partition_type balanced_and_unbalanced --alpha 5 --seed 41 --wandb True --target 0.1 --sweep True --noise_multiplier 0 --clipping 100000 --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode disparity --percentage_unbalanced_nodes 0.5 --unbalanced_ratio 0.5


# Node has 824 -  Counter({(0, 1): 283, (0, 0): 228, (1, 0): 212, (1, 1): 101})
# Node has 824 -  Counter({(0, 1): 182, (0, 0): 228, (1, 0): 212, , (1, 1): 202})   