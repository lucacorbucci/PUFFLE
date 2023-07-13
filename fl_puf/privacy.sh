
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --noise_multiplier 2 --delta 1e-6 &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --noise_multiplier 5 --delta 1e-6 &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --noise_multiplier 10 --delta 1e-6 &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --noise_multiplier 2 --delta 1e-6 &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --noise_multiplier 5 --delta 1e-6 &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --noise_multiplier 10 --delta 1e-6 &
wait;