

# # tunable_0.05
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.05_dp_5 --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=1.3949910507341277 --batch_size=442 --clipping=9.743360130373777 --epochs=5 --lr=0.04512719130141951 --momentum=0.1379172460836363 --optimizer=sgd --weight_decay_lambda=0.46141849158847775 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 

# # tunable_0.08
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.08_dp_5 --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=2.7140607570562256 --batch_size=442 --clipping=13.915890773121957 --epochs=4 --lr=0.053438083447466686 --momentum=0.8138296979942686 --optimizer=sgd --weight_decay_lambda=0.15350588333885695 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 


# # tunable_0.17
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.17_dp_5 --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=0.3757925224316564 --batch_size=377 --clipping=4.507692325642721 --epochs=4 --lr=0.02769884841050519 --momentum=0.8551017145229566 --optimizer=adam --weight_decay_lambda=0.8022682161174476 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 

# # tunable_0.20
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.20_dp_5 --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=0.9135722207773092 --batch_size=345 --clipping=7.569058844217375 --epochs=5 --lr=0.02061835503221202 --momentum=0.20987801763273192 --optimizer=adam --weight_decay_lambda=0.18425215735100045 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 


# # tunable_0.20_NO_DP
# for i in $(seq 0 5);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.20_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.312352208545676 --batch_size=334 --epochs=5 --lr=0.09857370540887776 --momentum=0.2507051742805466 --optimizer=sgd --weight_decay_lambda=0.6067019753311191 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --metric error_rate --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 


# # fixed_017
# for i in $(seq 0 10);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name fixed_017_epsilon_5 --project_name celeba_journal --node_shuffle_seed $i --batch_size=399 --clipping=1.1640496098441766 --epochs=4 --lr=0.037799811995181246 --optimizer=adam --regularization_lambda=0.6342039103686573 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --epsilon 4.5 --epsilon_statistics 0.5 --splitted_data_dir federated_2 --privileged_group 3 4 --unprivileged_group 2 1 0 --regularization_mode fixed --regularization True --target 0.17
# done 




for i in $(seq 10 12);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name baseline --project_name celeba_journal --node_shuffle_seed $i --batch_size=406 --epochs=5 --lr=0.003003163028416238 --node_shuffle_seed=908654067 --optimizer=adam --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.335 --sampled_clients_test 1 --sampled_clients_validation 0 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --validation_nodes 0 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated_2 --privileged_group 3 4 --unprivileged_group 2 1 0
done 