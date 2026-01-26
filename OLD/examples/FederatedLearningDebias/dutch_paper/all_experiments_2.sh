

# tunable_0.17_error_rate_50_dp_epsilon_05
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_dp_05 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=1.1750722968009109 --batch_size=841 --clipping=2.233728923683312 --epochs=4 --lr=0.08546696530278378 --momentum=0.4188188731186018 --optimizer=sgd --weight_decay_lambda=0.6173453496420496 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True
done

# tunable_0.17_error_rate_50_dp
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_dp_1 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=1.1089991066694702 --batch_size=747 --clipping=2.2184097583023568 --epochs=4 --lr=0.05701695451436958 --momentum=0.3200010501503857 --optimizer=sgd --weight_decay_lambda=0.3752285185080297 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True
done

# tunable_0.17_error_rate_50_NO_DP
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_NO_DP --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=0.5659943394093943 --batch_size=671 --epochs=3 --lr=0.08410114740066317 --momentum=0.6443016949671911 --optimizer=sgd --weight_decay_lambda=0.3081093166714726 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True --one_group_nodes True
done




# tunable_0.20_error_rate_50_dp_epsilon_05
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_dp_05 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=3.5723311195389216 --batch_size=632 --clipping=1.0615622302823895 --epochs=4 --lr=0.08227445110399514 --momentum=0.8093645948074776 --optimizer=adam --weight_decay_lambda=0.17732732333507145 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True
done

# tunable_0.20_error_rate_50_dp
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_dp_1 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=0.49191424995059785 --batch_size=446 --clipping=1.9298979207352391 --epochs=5 --lr=0.062124451738790616 --momentum=0.08556273418003166 --optimizer=sgd --weight_decay_lambda=0.41461749362241856 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True
done

# tunable_0.20_error_rate_50_NO_DP
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_NO_DP --project_name dutch_journal --node_shuffle_seed $i --alpha_target_lambda=0.6253848678332128 --batch_size=940 --epochs=5 --lr=0.07033976450956077 --momentum=0.4004735925120331 --optimizer=sgd --weight_decay_lambda=0.8520368154696548 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True --one_group_nodes True
done