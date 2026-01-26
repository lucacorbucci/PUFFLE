
# # fixed_005_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_005_NO_DP --node_shuffle_seed $i --project_name Continual_Income  --batch_size=1633 --epochs=3 --lr=0.08355285382170677 --optimizer=adam --regularization_lambda=0.7578481222626425 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
# done


# # fixed_01_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_01_NO_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=3453 --epochs=5 --lr=0.08449185910495356 --optimizer=adam --regularization_lambda=0.4133472834096977 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1
# done


# # fixed_015_NO_DP
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_015_NO_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=4823 --epochs=2 --lr=0.06118814866662815 --optimizer=adam --regularization_lambda=0.16860790329429234 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.15
# done



# # tunable_015
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py  --run_name tunable_015_DP_1 --node_shuffle_seed $i --project_name Continual_Income --alpha_target_lambda=1.850038860196807 --batch_size=4937 --clipping=5.96601310872564 --epochs=5 --lr=0.07335109273706576 --momentum=0.35620160881869917 --optimizer=adam --weight_decay_lambda=0.0032375898570655404 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 5 --ratio_unfair_nodes 0.5 --epsilon 0.9 --epsilon_lambda 0.1 --regularization_mode tunable --regularization True --target 0.15
# done



#  tunable_005_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --alpha_target_lambda=3.302304406183326 --run_name tunable_005_NO_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=3660 --epochs=4 --lr=0.08516585859862384 --momentum=0.3659871357559097 --optimizer=adam --weight_decay_lambda=0.9693293947950516 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done




