

# tunable_005_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py  --run_name tunable_005_NO_DP --node_shuffle_seed $i --project_name Continual_Income --alpha_target_lambda=3.961891796619548 --batch_size=4835 --epochs=2 --lr=0.07566538292264405 --momentum=0.4567221942634781 --optimizer=adam --weight_decay_lambda=0.6537489356990337 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done


# tunable_01_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name tunable_01_NO_DP --node_shuffle_seed $i --project_name Continual_Income  --alpha_target_lambda=2.4687720058783733 --batch_size=2739 --epochs=5 --lr=0.007528797630143794 --momentum=0.48235146031023896 --optimizer=adam --weight_decay_lambda=0.6585303139262145 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1
done


# tunable_015_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name tunable_015_NO_DP --node_shuffle_seed $i --project_name Continual_Income  --alpha_target_lambda=2.572579240243767 --batch_size=3901 --epochs=5 --lr=0.03267509527015567 --momentum=0.6131627769014412 --optimizer=adam --weight_decay_lambda=0.34682341076776385 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.15
done


# fixed_005_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_005_NO_DP --node_shuffle_seed $i --project_name Continual_Income  --batch_size=1633 --epochs=3 --lr=0.08355285382170677 --optimizer=adam --regularization_lambda=0.7578481222626425 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done


# fixed_01_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_01_NO_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=3453 --epochs=5 --lr=0.08449185910495356 --optimizer=adam --regularization_lambda=0.4133472834096977 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1
done


# fixed_015_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_015_NO_DP --node_shuffle_seed $i --project_name Continual_Income --batch_size=4823 --epochs=2 --lr=0.06118814866662815 --optimizer=adam --regularization_lambda=0.16860790329429234 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.15
done