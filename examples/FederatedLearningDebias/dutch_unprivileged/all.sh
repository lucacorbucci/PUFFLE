# tunable_0.05_error_rate_50_NO_DP
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_NO_DP --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=2.2221172785063734 --batch_size=985 --epochs=5 --lr=0.09953517204416208 --momentum=0.4937404201459936 --optimizer=sgd --weight_decay_lambda=0.9766075716114958 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done 

# tunable_0.08_error_rate_50_NO_DP
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_NO_DP --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=2.8888035958402645 --batch_size=694 --epochs=5 --lr=0.08634475466524415 --momentum=0.17044848459147488 --optimizer=sgd --weight_decay_lambda=0.4781896069388488 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done 

# tunable_0.11_error_rate_50_NO_DP
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_NO_DP --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=3.030291838690565 --batch_size=482 --epochs=4 --lr=0.07719054561873387 --momentum=0.665245544169215 --optimizer=sgd --weight_decay_lambda=0.9160287041134516 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done 

# tunable_0.05_error_rate_50_dp
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_dp --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=3.2098261254024214 --batch_size=759 --clipping=1.8759406111032983 --epochs=4 --lr=0.03218978225738871 --momentum=0.2297427813020876 --optimizer=adam --weight_decay_lambda=0.6488566886662545 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done 

# tunable_0.05_error_rate_50_dp_epsilon_05
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.05_error_rate_50_dp_epsilon_05 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.387495162092908 --batch_size=624 --clipping=4.4857191281599125 --epochs=4 --lr=0.0805002621373067 --momentum=0.8716661291999491 --optimizer=sgd --weight_decay_lambda=0.810962670008903 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done 

# tunable_0.08_error_rate_50_dp_epsilon_05
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_dp_epsilon_05 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.7834210596019044 --batch_size=828 --clipping=1.1259713064805292 --epochs=5 --lr=0.0909876928649309 --momentum=0.8004722272762427 --optimizer=sgd --weight_decay_lambda=0.1939846945062876 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done 

# tunable_0.08_error_rate_50_dp
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.08_error_rate_50_dp --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.2193258057384115 --batch_size=560 --clipping=14.611042134299534 --epochs=2 --lr=0.09782671800869996 --momentum=0.17552826275814767 --optimizer=sgd --weight_decay_lambda=0.4779891537247008 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --global_computation True
done 







# tunable_0.11_error_rate_50_dp
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_dp --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=2.3928303683392578 --batch_size=832 --clipping=5.449186018687835 --epochs=5 --lr=0.0889953007296564 --momentum=0.6889208893218906 --optimizer=sgd --weight_decay_lambda=0.5066622830630502 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done 

# tunable_0.11_error_rate_50_dp_epsilon_05
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.11_error_rate_50_dp_epsilon_05 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=2.890156113182023 --batch_size=801 --clipping=2.3500154817883523 --epochs=3 --lr=0.09410749357723636 --momentum=0.5925563786137372 --optimizer=sgd --weight_decay_lambda=0.22008049829828363 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --global_computation True
done 

# tunable_0.17_error_rate_50_dp_epsilon_05
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_dp_epsilon_05 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.1750722968009109 --batch_size=841 --clipping=2.233728923683312 --epochs=4 --lr=0.08546696530278378 --momentum=0.4188188731186018 --optimizer=sgd --weight_decay_lambda=0.6173453496420496 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True
done 

# tunable_0.20_error_rate_50_dp_epsilon_05
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_dp_epsilon_05 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=3.5723311195389216 --batch_size=632 --clipping=1.0615622302823895 --epochs=4 --lr=0.08227445110399514 --momentum=0.8093645948074776 --optimizer=adam --weight_decay_lambda=0.17732732333507145 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.3 --epsilon_statistics 0.1 --epsilon_lambda 0.1 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True
done 

# tunable_0.17_error_rate_50_dp
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_dp --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.1089991066694702 --batch_size=747 --clipping=2.2184097583023568 --epochs=4 --lr=0.05701695451436958 --momentum=0.3200010501503857 --optimizer=sgd --weight_decay_lambda=0.3752285185080297 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True
done 

# tunable_0.20_error_rate_50_dp
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_dp --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=0.49191424995059785 --batch_size=446 --clipping=1.9298979207352391 --epochs=5 --lr=0.062124451738790616 --momentum=0.08556273418003166 --optimizer=sgd --weight_decay_lambda=0.41461749362241856 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --one_group_nodes True --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --epsilon 0.5 --epsilon_statistics 0.1 --epsilon_lambda 0.4 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True
done 

# tunable_0.17_error_rate_50_NO_DP
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.17_error_rate_50_NO_DP --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=0.5659943394093943 --batch_size=671 --epochs=3 --lr=0.08410114740066317 --momentum=0.6443016949671911 --optimizer=sgd --weight_decay_lambda=0.3081093166714726 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --global_computation True --one_group_nodes True
done 

# tunable_0.20_error_rate_50_NO_DP
for i in $(seq 0 8);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/dutch_unprivileged/../main.py --run_name tunable_0.20_error_rate_50_NO_DP --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=0.6253848678332128 --batch_size=940 --epochs=5 --lr=0.07033976450956077 --momentum=0.4004735925120331 --optimizer=sgd --weight_decay_lambda=0.8520368154696548 --dataset dutch_unprivileged --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/dutch/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --dataset_path ../../../../data/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.6 0.8 --approach representative --splitted_data_dir federated_2 --metric error_rate --unprivileged_group 3 4 --privileged_group 1 2 5 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --global_computation True --one_group_nodes True
done 




