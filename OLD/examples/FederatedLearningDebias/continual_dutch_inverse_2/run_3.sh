
# for i in $(seq 4 6);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Fixed_Inverse_Tunable_Mid_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=39 --epochs=7 --lr=0.05748110547350514 --optimizer=sgd --regularization_lambda=0.2623662469158684 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done

# # Baseline

# for i in $(seq 4 6);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --batch_size=91 --epochs=3 --lr=0.06516279258593441 --optimizer=adam --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5
# done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --batch_size=91 --run_name Baseline_Inverse_Mid_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --epochs=3 --lr=0.06516279258593441 --optimizer=adam --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5
done