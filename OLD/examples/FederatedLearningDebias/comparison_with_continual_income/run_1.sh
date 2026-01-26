# # Baseline 
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_continual_income/../main.py --run_name No_shift_Baseline_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=4628 --epochs=5 --lr=0.09126801823868474 --optimizer=adam --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5
# done


# Baseline_DP_1
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison with continual_income/../main.py --batch_size=4628 --epochs=5 --lr=0.09126801823868474 --optimizer=adam --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.3 --sampled_clients_test 0 --sampled_clients_validation 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --sweep True --training_nodes 0.4 --validation_nodes 0.27 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --ratio_unfair_nodes 0.5
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_continual_income/../main.py --run_name No_shift_Baseline_DP_1 --node_shuffle_seed $i --project_name Continual_Income --batch_size=3625 --clipping=5.838304223548287 --epochs=5 --lr=0.03637353229256966 --optimizer=adam --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 5 --epsilon 1
done
