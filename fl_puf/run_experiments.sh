poetry run main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --alpha 1000000000 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --DPL True --DPL_lambda 0.2 &
wait;
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --alpha 1000000000 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --private True --epsilon 10 --delta 1e-6 --clipping 10.0 &
wait;
poetry run python main.py --num_rounds 10 --dataset celeba --epochs 1 --num_client_cpus 1 --alpha 1000000000 --num_client_gpus 1 --batch_size 512 --pool_size 1 --sampled_clients 1.0 --lr 0.01 --train_csv train_original --test_csv test_original --debug True --base_path ../data --wandb True --private True --epsilon 10 --delta 1e-6 --clipping 10.0 --DPL True --DPL_lambda 0.2 &
wait;
