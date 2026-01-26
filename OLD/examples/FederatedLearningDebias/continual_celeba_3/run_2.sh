

for i in $(seq 43 48);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_4/../main.py --run_name Fixed_06 --project_name Celeba_continual_Results --node_shuffle_seed $i --batch_size=374 --clipping=6.038495379956827 --epochs=4 --lr=0.039940195923251806 --optimizer=adam --regularization_lambda=0.570615709744962 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_06 --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_2 --shift True --switch_dataset 21 --regularization_mode fixed --regularization True --target 0.05 --epsilon 4.5 --epsilon_statistics 0.5
done


for i in $(seq 50 55);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_celeba_2/../main.py --run_name Tunable_Mid --project_name Celeba_continual_Results --node_shuffle_seed $i --alpha_target_lambda=2.9810589674494268 --batch_size=307 --clipping=5.467132378765513 --epochs=2 --lr=0.01842122656307251 --momentum=0.004204435778382943 --optimizer=adam --weight_decay_lambda=0.9670456968831976 --dataset celeba --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --train_csv df_attractive_mid --train_csv_shift df_not_attractive --debug False --base_path ../../../../data/ --dataset_path ../../../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 316 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --number_of_samples_per_node_shift 800 --ratio_unfair_nodes_shift 0.5 --ratio_unfairness_shift 0 0 --splitted_data_dir federated_2 --shift True --switch_dataset 21 --regularization_mode tunable --regularization True --target 0.05 --global_computation True --epsilon 3.5 --epsilon_statistics 0.5 --epsilon_lambda 1
done 

