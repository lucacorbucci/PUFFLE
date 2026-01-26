
# # fixed_0.08_error_rate_50_dp_epsilon_2
# for i in $(seq 10 16);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name fixed_0.08_error_rate_50_dp_epsilon_2 --project_name dutch_journal --batch_size=860 --clipping=18.723982353623175 --epochs=3 --lr=0.05146668156891875 --node_shuffle_seed=$i --optimizer=adam --regularization_lambda=0.22501228316541955 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_3 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 1.8 --epsilon_statistics 0.2 --regularization_mode fixed --regularization True --target 0.08
# done

# # tunable_0.05_error_rate_50_dp_epsilon_2
# for i in $(seq 10 16);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_dp_epsilon_2 --project_name dutch_journal --alpha_target_lambda=0.32332924026212434 --batch_size=470 --clipping=1.51175702749221 --epochs=2 --lr=0.09059610672834646 --momentum=0.3388605875953432 --node_shuffle_seed=$i --optimizer=sgd --weight_decay_lambda=0.31705035295991935 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_3 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.8 --epsilon_statistics 0.2 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.05
# done

# # tunable_0.08_error_rate_50_dp_epsilon_5
# for i in $(seq 10 16);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_dp_epsilon_5 --project_name dutch_journal --alpha_target_lambda=0.8535659614196875 --batch_size=885 --clipping=11.77355304420882 --epochs=4 --lr=0.09062768024160626 --momentum=0.5829345006485905 --node_shuffle_seed=$i --optimizer=adam --weight_decay_lambda=0.11368544041992493 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_3 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.8 --epsilon_statistics 0.2 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.08
# done



