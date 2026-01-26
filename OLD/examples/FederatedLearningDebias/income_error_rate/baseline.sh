
for i in $(seq 111 118);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/income_error_rate/../main.py --run_name baseline_dp_1.0 --project_name income_journal --node_shuffle_seed $i --batch_size=9176 --clipping=11.180837275874822 --epochs=5 --lr=0.08805618090220714 --optimizer=adam --dataset income --num_rounds 10 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 48 --sampled_clients 0.29 --sampled_clients_test 1 --debug False --base_path ../../../../data/income_error_rate/ --dataset_path ../../../../data/income_error_rate/ --seed 42 --wandb True --training_nodes 0.81 --test_nodes 0.2 --tabular_data True --metric error_rate --epsilon 1 --splitted_data_dir federated --privileged_group 1 2 3 4 --unprivileged_group 5 6 7 8 9
done