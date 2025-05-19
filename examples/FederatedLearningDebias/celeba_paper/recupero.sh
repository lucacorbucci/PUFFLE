
# tunable_dp_5_t_0.09
for i in $(seq 10 25);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_paper/../main.py --run_name tunable_dp_5_t_0.09 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=3.0946695469306693 --batch_size=65 --clipping=2.734214516379665 --epochs=4 --lr=0.028921397536495124 --momentum=0.31047582008851243 --optimizer=adam --weight_decay_lambda=0.4491910929641054 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv original_merged --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated_3 --metric disparity --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.09 --global_computation True
done 


# tunable_dp_5_t_0.06
for i in $(seq 10 20);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/celeba_paper/../main.py --run_name tunable_dp_5_t_0.06 --project_name celeba_plots_paper --node_shuffle_seed $i --alpha_target_lambda=1.2368096281642236 --batch_size=446 --clipping=13.223996693028996 --epochs=5 --lr=0.06393490809974926 --momentum=0.35478492233833553 --optimizer=sgd --weight_decay_lambda=0.10568050047418144 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv original_merged --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated_3 --metric disparity --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --update_lambda True --regularization_mode tunable --regularization True --target 0.06 --global_computation True
done 
