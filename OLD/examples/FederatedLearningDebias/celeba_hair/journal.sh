



# # tunable_0.05_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.05_NO_DP --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=0.5012010591311633 --batch_size=509 --epochs=1 --lr=0.009100189941423096 --momentum=0.1705720039399954 --optimizer=adam --weight_decay_lambda=0.015285739327443426 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done

# # tunable_0.08_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.08_NO_DP --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=3.958373751986107 --batch_size=419 --epochs=5 --lr=0.0852468972159736 --momentum=0.06280940521724426 --optimizer=sgd --weight_decay_lambda=0.8703889596696662 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.08 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 

# # tunable_0.11_NO_DP
# for i in $(seq 0 9);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.11_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.772560443532443 --batch_size=363 --epochs=5 --lr=0.09406266921309486 --momentum=0.327426995819408 --optimizer=sgd --weight_decay_lambda=0.7584250944348248 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.11 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 

# tunable_0.17_NO_DP
# for i in $(seq 0 15);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.05_NO_DP --project_name celeba_journal --node_shuffle_seed $i /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --alpha_target_lambda=3.9337388267000057 --batch_size=268 --epochs=5 --lr=0.09869574843590904 --momentum=0.4098902029529703 --optimizer=sgd --weight_decay_lambda=0.744137153812604 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.05 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done

# # tunable_0.20_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.20_NO_DP --project_name celeba_journal --node_shuffle_seed $i  --alpha_target_lambda=1.6914538659620362 --batch_size=419 --epochs=5 --lr=0.09356246104961902 --momentum=0.059322169338138254 --optimizer=sgd --weight_decay_lambda=0.09673488928074137 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done

# for i in $(seq 0 10);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.20_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.312352208545676 --batch_size=334 --epochs=5 --lr=0.09857370540887776 --momentum=0.2507051742805466 --optimizer=sgd --weight_decay_lambda=0.6067019753311191 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --sampled_clients_validation 0 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --metric error_rate --splitted_data_dir federated_2 --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 


# for i in $(seq 0 10);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.17_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.9848599269558114 --batch_size=416 --epochs=5 --lr=0.09566319185155968 --momentum=0.05886964580601124 --optimizer=sgd --weight_decay_lambda=0.045148776037402936 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --sampled_clients_validation 0 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 


# tunable_0.17_NO_DP
# for i in $(seq 0 10);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.17_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.9848599269558114 --batch_size=416 --epochs=5 --lr=0.09566319185155968 --momentum=0.05886964580601124 --optimizer=sgd --weight_decay_lambda=0.045148776037402936 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.17 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 

# # tunable_0.20_NO_DP
# for i in $(seq 0 5);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name tunable_0.20_NO_DP --project_name celeba_journal --node_shuffle_seed $i --alpha_target_lambda=2.312352208545676 --batch_size=334 --epochs=5 --lr=0.09857370540887776 --momentum=0.2507051742805466 --optimizer=sgd --weight_decay_lambda=0.6067019753311191 --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --metric error_rate --splitted_data_dir federated --update_lambda True --regularization_mode tunable --regularization True --target 0.2 --privileged_group 3 4 --unprivileged_group 2 1 0 --global_computation True
# done 



for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_hair/../main.py --run_name baseline --project_name celeba_journal --node_shuffle_seed $i --batch_size=406 --epochs=5 --lr=0.003003163028416238 --node_shuffle_seed=908654067 --optimizer=adam --dataset celeba_hair --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.335 --sampled_clients_test 1 --sampled_clients_validation 0 --train_csv attractive --debug False --base_path ../../../../data --dataset_path ../../../../data/celeba_hair/ --seed 41 --wandb True --training_nodes 0.67 --validation_nodes 0 --test_nodes 0.335 --group_to_reduce 1 1 --group_to_increment 0 1 --partition_type representative --number_of_samples_per_node 830 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --metric error_rate --splitted_data_dir federated --privileged_group 3 4 --unprivileged_group 2 1 0
done 