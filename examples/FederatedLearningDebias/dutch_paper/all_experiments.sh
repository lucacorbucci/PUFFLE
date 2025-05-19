
# tunable_0.05_error_rate_50_dp
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_dp_1 --project_name dutch_journal --node_shuffle_seed $i --alpha_target_lambda=3.6678248352984415 --batch_size=791 --clipping=1.416494267871773 --epochs=4 --lr=0.05300018680001619 --momentum=0.5472104902358474 --optimizer=adam --weight_decay_lambda=0.7705738624581367 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done

# tunable_0.05_error_rate_50_NO_DP
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_NO_DP --project_name dutch_journal --node_shuffle_seed $i --alpha_target_lambda=2.2221172785063734 --batch_size=985 --epochs=5 --lr=0.09953517204416208 --momentum=0.4937404201459936 --optimizer=sgd --weight_decay_lambda=0.9766075716114958 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done

# tunable_0.05_error_rate_50_dp_epsilon_05
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_dp_05 --project_name dutch_journal --node_shuffle_seed $i --alpha_target_lambda=1.387495162092908 --batch_size=624 --clipping=4.4857191281599125 --epochs=4 --lr=0.0805002621373067 --momentum=0.8716661291999491 --optimizer=sgd --weight_decay_lambda=0.810962670008903 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done




# tunable_0.08_error_rate_50_NO_DP
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_NO_DP --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=2.8888035958402645 --batch_size=694 --epochs=5 --lr=0.08634475466524415 --momentum=0.17044848459147488 --optimizer=sgd --weight_decay_lambda=0.4781896069388488 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done

# tunable_0.08_error_rate_50_dp_epsilon_05
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_dp_05 --project_name dutch_journal --node_shuffle_seed $i --alpha_target_lambda=1.7834210596019044 --batch_size=828 --clipping=1.1259713064805292 --epochs=5 --lr=0.0909876928649309 --momentum=0.8004722272762427 --optimizer=sgd --weight_decay_lambda=0.1939846945062876 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done

# tunable_0.08_error_rate_50_dp
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_dp_1 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=1.2193258057384115 --batch_size=560 --clipping=14.611042134299534 --epochs=2 --lr=0.09782671800869996 --momentum=0.17552826275814767 --optimizer=sgd --weight_decay_lambda=0.4779891537247008 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done




# tunable_0.11_error_rate_50_NO_DP
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_NO_DP --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=3.030291838690565 --batch_size=482 --epochs=4 --lr=0.07719054561873387 --momentum=0.665245544169215 --optimizer=sgd --weight_decay_lambda=0.9160287041134516 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done

# tunable_0.11_error_rate_50_dp
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_dp_1 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=2.3928303683392578 --batch_size=832 --clipping=5.449186018687835 --epochs=5 --lr=0.0889953007296564 --momentum=0.6889208893218906 --optimizer=sgd --weight_decay_lambda=0.5066622830630502 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done

# tunable_0.11_error_rate_50_dp_epsilon_05
for i in $(seq 230 234);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_dp_05 --project_name dutch_journal --node_shuffle_seed $i  --alpha_target_lambda=2.890156113182023 --batch_size=801 --clipping=2.3500154817883523 --epochs=3 --lr=0.09410749357723636 --momentum=0.5925563786137372 --optimizer=sgd --weight_decay_lambda=0.22008049829828363 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done


