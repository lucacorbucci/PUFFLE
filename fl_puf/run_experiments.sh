# python main.py --num_rounds 20 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1 --lr 0.01 --wandb True

# python main.py --num_rounds 20 --dataset celeba --epochs 1 --batch_size 512 --pool_size 1 --sampled_clients 1 --lr 0.01 --wandb True --DPL True --DPL_lambda 2.0

python main.py --num_rounds 20 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 99 --sampled_clients 0.1 --lr 0.01 --wandb True

python main.py --num_rounds 20 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 99 --sampled_clients 0.1 --lr 0.01 --wandb True --DPL True --DPL_lambda 2.0