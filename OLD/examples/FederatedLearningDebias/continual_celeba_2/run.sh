# MID 

# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_2/../main.py --run_name Baseline_MID --project_name Celeba_continual_Results --node_shuffle_seed $i  --batch_size=62 --clipping=6.306988676780853 --epochs=3 --lr=0.08725402608297399 --optimizer=sgd --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_mid --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated --shift True --switch_dataset 21 --epsilon 5
# done

# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_2/../main.py --run_name Fixed_MID --project_name Celeba_continual_Results --node_shuffle_seed $i  --batch_size=218 --clipping=5.166692076624496 --epochs=5 --lr=0.05958506062696869 --optimizer=sgd --regularization_lambda=0.10108912693312493 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_mid --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated --shift True --switch_dataset 21 --regularization_mode fixed --regularization True --target 0.05 --epsilon 4.5 --epsilon_statistics 0.5
# done

# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_2/../main.py --run_name Tunable_MID --project_name Celeba_continual_Results --node_shuffle_seed $i  --alpha_target_lambda=3.493361304437932 --batch_size=442 --clipping=8.23182167563376 --epochs=3 --lr=0.03102069996418947 --momentum=0.732222056172896 --optimizer=adam --weight_decay_lambda=0.18400347396925232 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_mid --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated --shift True --switch_dataset 21 --regularization_mode tunable --regularization True --target 0.05 --global_computation True --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1
# done


# Tunable DP 3
for i in $(seq 50 54);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_4/../main.py --run_name Tunable_05  --project_name Celeba_continual_Results --node_shuffle_seed $i --alpha_target_lambda=2.008186651569194 --batch_size=176 --clipping=11.404641444217257 --epochs=5 --lr=0.02072824257086644 --momentum=0.019318993575501733 --optimizer=adam --weight_decay_lambda=0.08217723498717906 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_06 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_4 --shift True --switch_dataset 21 --regularization_mode tunable --regularization True --target 0.05 --global_computation True --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1
done

# poetry run wandb agent lucacorbucci/Celeba_continual/du6legnx

# # Tunable DP 4poetry run wandb agent lucacorbucci/Celeba_continual/du6legnx
# for i in $(seq 77 81);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_2/../main.py --run_name Tunable_MID  --project_name Celeba_continual_Results --node_shuffle_seed $i --alpha_target_lambda=0.32044622885366003 --batch_size=72 --clipping=9.08838990572924 --epochs=5 --lr=0.03011288644384641 --momentum=0.18272307428666903 --optimizer=adam --weight_decay_lambda=0.6774016413653103 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_05 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_2 --shift True --switch_dataset 21 --regularization_mode tunable --regularization True --target 0.05 --global_computation True --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1
# done



# SWEEP DA RIAVVIARE


# tunable_DP_4
# federated_3
# poetry run wandb agent lucacorbucci/Celeba_continual/a45zcho3 --count 60


# poetry run wandb agent lucacorbucci/Celeba_continual/ev773vj0 --count 60
