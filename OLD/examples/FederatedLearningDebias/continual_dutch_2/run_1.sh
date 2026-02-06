# # dutch 2

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py --run_name Fixed_2_no_DP --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=111 --epochs=3 --lr=0.03687835913515573 --optimizer=adam --regularization_lambda=0.8773406169997107 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py --run_name Fixed_2_no_DP --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=120 --epochs=3 --lr=0.05447893443586349 --optimizer=adam --regularization_lambda=0.716646563835218 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py --run_name Fixed_2_no_DP --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=118 --epochs=4 --lr=0.021374782119416146 --optimizer=adam --regularization_lambda=0.4319573527217644 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py --run_name Fixed_2_no_DP --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=66 --epochs=2 --lr=0.0782892858713795 --optimizer=adam --regularization_lambda=0.8821096411432686 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py --run_name Tunable_dutch_3_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.3909306185594969 --batch_size=114 --epochs=4 --lr=0.07518553543857778 --momentum=0.7001434440400687 --optimizer=sgd --weight_decay_lambda=0.2426041215461616 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_2/../main.py  --run_name Tunable_dutch_3_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.1550410053859186 --batch_size=102 --epochs=3 --lr=0.07645949369009666 --momentum=0.6291380734096034 --optimizer=sgd --weight_decay_lambda=0.4034275154183937 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_2/ --dataset_path ../../../../data/continual_dutch_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

