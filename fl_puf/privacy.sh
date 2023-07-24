# Baseline without privacy
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 &
wait;

# Private
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 8 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --noise_multiplier 2 --delta 1e-4 &
wait;

# Private + DPL
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 8 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 5 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 --private True --clipping 10 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 --private True --clipping 8 --noise_multiplier 2 --delta 1e-4 &
wait;

poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 --private True --clipping 5 --noise_multiplier 2 --delta 1e-4 &
wait;






# DPL 0.1
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.1 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 &
wait;

# DPL 0.2
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 &
wait;

# DPL 0.3
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.3 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 &
wait;

# DPL 0.4
poetry run python main.py --num_rounds 20 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 &
wait;

