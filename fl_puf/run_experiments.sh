# # Baseline with 1 client
# # poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# # wait;
# # poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# # wait;
# # poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --private True --epsilon 10 --delta 1e-6 --clipping 10.0 &
# # wait;
# # poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --private True --epsilon 10 --delta 1e-6 --clipping 10.0 --DPL True --DPL_lambda 0.2 &
# # wait;


# # # Baseline with 100 clients
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# # wait;
# # # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# # wait;
# # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.05 &
# # wait;
# # # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.1 &
# # wait;



# # # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# # wait;
# # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.05 &
# # wait;
# # # Baseline with 100 clients and DPL
# # poetry run python main.py --num_rounds 20 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.1 &
# # wait;

# # Baseline with 100 clients
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.7 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.6 &
# wait;
# # Baseline with 100 clients and DPL
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.7 &
# wait;


# Baseline 100 clients con lr 0.1
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.3 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.6 &
# wait;



# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.3 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.4 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.6 &
# wait;


poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --epsilon 10 --delta 1e-6   &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 2 --epsilon 10 --delta 1e-6   &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --epsilon 5 --delta 1e-6   &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --epsilon 10 --delta 1e-6   &
wait;


poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 5 --epsilon 5 --delta 1e-6   &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 2 --epsilon 5 --delta 1e-6   &
wait;

poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 1 --epsilon 5 --delta 1e-6   &
wait;


poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 1 --epsilon 10 --delta 1e-6   &
wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 5 --epsilon 10 --delta 1e-6   &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 --private True --clipping 10 --epsilon 10 --delta 1e-6   &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 --private True --clipping 10 --epsilon 10 --delta 1e-6   &
# wait;




# Baseline 100 clients con lr 0.01 4 volte per vedere se sono uguali i risultati
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;


## Esperimenti da aggiungere
# Batch size più piccolo con learning rate 0.01 e 0.1

# # Batch size 256 with LR 0.1 and 0.01
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;



# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;



# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 8 --num_client_cpus 1 --num_client_gpus 1 --batch_size 128 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;


# 20 Local epchs with LR 0.1 and 0.01, batch size 512
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 512 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;


# Batch size 256 with LR 0.1 and 0.01, 20 local epochs
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.5 &
# wait;

# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;
# poetry run python main.py --num_rounds 10 --dataset celeba --epochs 20 --num_client_cpus 1 --num_client_gpus 1 --batch_size 256 --pool_size 100 --sampled_clients 0.1 --lr 0.1 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.8 &
# wait;