# Baseline without privacy
poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --wandb True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --alpha 1 &
wait;
# Baseline without privacy
poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --wandb True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --alpha 5 &
wait;
# Baseline without privacy
poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --wandb True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --alpha 10 &
wait;
# Baseline without privacy
poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --wandb True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --alpha 100 &
wait;
# Baseline without privacy
poetry run python main.py --num_rounds 30 --dataset celeba --epochs 4 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --DPL True --DPL_lambda 0 --wandb True --private True --clipping 10000000000 --noise_multiplier 0 --delta 1e-4 --alpha 1000 &
wait;
