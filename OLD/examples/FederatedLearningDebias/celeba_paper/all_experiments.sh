# fixed_t_0.06
for i in $(seq 10 20);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_paper/../main.py --run_name fixed_t_0.06 --project_name celeba_plots_paper --node_shuffle_seed $i --batch_size=337 --epochs=3 --lr=0.06764223063121005 --optimizer=sgd --regularization_lambda=0.4928382970016825 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv original_merged --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated_2 --metric disparity --update_lambda False --regularization_mode fixed --regularization True --target 0.06
done 

# tunable_t_0.04
for i in $(seq 10 20);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_paper/../main.py --run_name tunable_t_0.04 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=2.7224227338142226 --batch_size=66 --epochs=5 --lr=0.0902564198603916 --momentum=0.38699870580315304 --optimizer=sgd --weight_decay_lambda=0.5815866285117238 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv original_merged --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated_2 --metric disparity --update_lambda True --regularization_mode tunable --regularization True --target 0.04 --global_computation True
done 

# tunable_dp_5_t_0.04
for i in $(seq 10 20);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_paper/../main.py --run_name tunable_dp_5_t_0.04 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=0.8024131825115238 --batch_size=223 --clipping=19.626810952630464 --epochs=3 --lr=0.06603503402505119 --momentum=0.6751751811978708 --optimizer=sgd --weight_decay_lambda=0.08655441312315527 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv original_merged --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated_2 --metric disparity --epsilon 4 --epsilon_statistics 0.5 --epsilon_lambda 0.5 --update_lambda True --regularization_mode tunable --regularization True --target 0.04 --global_computation True
done 




