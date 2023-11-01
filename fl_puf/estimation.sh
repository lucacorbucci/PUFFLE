# # IID distribution

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type iid --alpha 5 &
# wait;

# # Non-IID Distribution

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 &
# wait;

# # Underrepresented Distribution (7 nodes with Female and 3 nodes with Male, with 10000 samples in Male, Smiling Group)

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;


# # Underrepresented Distribution (7 nodes with Female and 3 nodes with Male, with 10000 samples in Male, Smiling Group)

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# # # DPL with prob. estimations
# # poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# # wait;

# # # Baseline
# # poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# # wait;

# # Unbalanced with 50% of the nodes with male samples and 50% with female samples


# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;


# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;





# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # DPL with prob. estimation
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;


# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;

# # DPL with prob. estimation
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;


# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;

# # DPL with prob. estimation
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;



# # Classic DPL
# poetry run python main.py --num_rounds 3 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 0.5 --lr 0.1 --train_csv small_sampled_train --test_csv small_sampled_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 1.0 &
# wait;

# # DPL with prob. estimation
# poetry run python main.py --num_rounds 3 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 0.5 --lr 0.1 --train_csv small_sampled_train --test_csv small_sampled_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 1.0 &
# wait;



# # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.8 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.8 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.8 &
# wait;






# # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_cliunbalancedent_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type  --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;




# da qui

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;



# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;



# # da qui in poi vanno eseguiti

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 40 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;


# # Unbalanced

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 &
# wait;



# # Underrepresented with higher Lambda

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;


# # Baseline
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;


# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;


# # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;



# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --percentage_unbalanced_nodes 0.3 &
# wait;

# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --percentage_unbalanced_nodes 0.3 &
# wait;

# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --percentage_unbalanced_nodes 0.3 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type iid &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type iid &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type iid &
# wait;


# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 &
# wait;







# # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 &
# wait;





# # DPL with prob. estimation
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;

# # Classic DPL
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;

# # Baseline
# poetry run python main.py --num_rounds 15 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type unbalanced_one_class --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.7 &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv small_sampled_train --test_csv small_sampled_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False --wandb True &
# wait;


# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 2 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 0.4 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv train_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &


# Unbalanced with 50% of the nodes with male samples and 50% with female samples

# DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;


# Underrepresented Distribution (7 nodes with Female and 3 nodes with Male, with 10000 samples in Male, Smiling Group)

# DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# Unbalanced with 50% of the nodes with male samples and 50% with female samples

# DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# Baseline
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;

# # DPL with prob. estimations
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.3 --sort_clients False &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.25 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --wandb True --seed 41 --percentage_unbalanced_nodes 0.5 --sort_clients False &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --wandb True --no_sort_clients  &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --wandb True --no_sort_clients  &
# wait;


# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --wandb True --no_sort_clients  &
# wait;

# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 80 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --wandb True --no_sort_clients  &
# wait;


# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --wandb True --no_sort_clients  &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --wandb True --no_sort_clients  &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv extreme --test_csv extreme_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv extreme --test_csv extreme_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.5 --wandb True --no_sort_clients  &
# wait;

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv extreme --test_csv extreme_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --no_sort_clients --wandb True  &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv extreme --test_csv extreme_test --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type underrepresented --alpha 5  --seed 41 --percentage_unbalanced_nodes 0.3 --wandb True --no_sort_clients  &
# wait;


# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --alpha 5 --seed 41 --percentage_unbalanced_nodes 0.3 --no_sort_clients --wandb True --alpha_target_lambda 0.01 --target 0.1

# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 --seed 41 --wandb True &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.01 --target 0.2 &
# wait;
# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train --test_csv unfair_test --debug True --base_path ../data --DPL True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.01 --target 0.3 &
# wait;


# poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type non_iid --alpha 5 --seed 41 --wandb True --alpha_target_lambda 0.01 --target 0.1 &



poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 1.0 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --probability_estimation True --partition_type underrepresented --percentage_unbalanced_nodes 0.3 --no_sort_clients --alpha 5 --seed 




poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 10 --sampled_clients 0.1 --lr 0.1 --train_csv unfair_train_reduced --test_csv unfair_test_reduced --debug True --base_path ../data --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --partition_type non_iid --alpha 5 --seed 41