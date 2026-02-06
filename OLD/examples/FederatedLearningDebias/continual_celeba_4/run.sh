# # 0.5
# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_4/../main.py --run_name Baseline_05 --project_name Celeba_continual_Results --node_shuffle_seed $i  --batch_size=357 --clipping=2.945566577144149 --epochs=5 --lr=0.04671789463108054 --optimizer=adam --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_05 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_2 --shift True --switch_dataset 21 --epsilon 5
# done 

# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_4/../main.py --run_name Fixed_05 --project_name Celeba_continual_Results --node_shuffle_seed $i  --batch_size=482 --clipping=6.385808461244725 --epochs=3 --lr=0.06999226614613444 --optimizer=adam --regularization_lambda=0.2547779265052374 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_05 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_3 --shift True --switch_dataset 21 --regularization_mode fixed --regularization True --target 0.05 --epsilon 4.5 --epsilon_statistics 0.5
# done 

# for i in $(seq 111 113);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_4/../main.py  --run_name Tunable_05 --project_name Celeba_continual_Results --node_shuffle_seed $i  --alpha_target_lambda=3.2541964774705328 --batch_size=265 --clipping=3.633567746188148 --epochs=4 --lr=0.09588008871150884 --momentum=0.14947757118749094 --optimizer=sgd --weight_decay_lambda=0.5160431039773479 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_05 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_3 --shift True --switch_dataset 21 --regularization_mode tunable --regularization True --target 0.05 --global_computation True --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1
# done




# tunable_DP_3
# federated 2
poetry run wandb agent lucacorbucci/Celeba_continual/9o95g6eq --count 60
 
# Fixed DP_2
# federated 2
poetry run wandb agent lucacorbucci/Celeba_continual/rcaemipe --count 60