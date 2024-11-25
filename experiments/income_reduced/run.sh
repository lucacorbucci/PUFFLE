# poetry run python /home/lcorbucci/multi_fairness/PUFFLE/experiments/income_reduced/../../puffle/main.py --project_name Multi_fairness_Results --run_name Baseline --batch_size=1226 --epochs=4 --lr=0.09579496880260412 --optimizer=sgd --dataset income --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --debug False --base_path ../../../reduced_income_data/ --dataset_path ../../../reduced_income_data/ --seed 41 --wandb True --training_nodes 1 --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --cross_silo True


for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/multi_fairness/PUFFLE/experiments/income_reduced/../../puffle/main.py --project_name Multi_fairness_Results --run_name Target_005 --node_shuffle_seed $i --batch_size=928 --epochs=5 --lr=0.09098362625211072 --optimizer=adam --regularization_lambda=0.596290699159395 --dataset income --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --debug False --base_path ../../../reduced_income_data/ --dataset_path ../../../reduced_income_data/ --seed 41 --wandb True --training_nodes 1  --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.05 --cross_silo True
done


for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/multi_fairness/PUFFLE/experiments/income_reduced/../../puffle/main.py --project_name Multi_fairness_Results --run_name Target_010 --node_shuffle_seed $i --batch_size=1358 --epochs=5 --lr=0.04181459868456289 --optimizer=adam --regularization_lambda=0.2527420787292905 --dataset income --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 20 --sampled_clients 1 --sampled_clients_test 1 --debug False --base_path ../../../reduced_income_data/ --dataset_path ../../../reduced_income_data/ --seed 41 --wandb True --training_nodes 1  --test_nodes 1 --tabular_data True --update_lambda False --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5 --update_lambda False --regularization_mode fixed --regularization True --target 0.1 --cross_silo True
done 