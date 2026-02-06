# Tunable

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Tunable_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.0606968629438187 --batch_size=128 --epochs=3 --lr=0.07688503638135313 --momentum=0.31058819562174994 --optimizer=sgd --weight_decay_lambda=0.2927189364111863 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Tunable_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.10655919977521304 --batch_size=99 --epochs=3 --lr=0.04287587390953909 --momentum=0.8321282120662346 --optimizer=sgd --weight_decay_lambda=0.3794232737812249 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Tunable_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.4434462445000216 --batch_size=104 --epochs=6 --lr=0.04277561293762476 --momentum=0.1482140501851187 --optimizer=sgd --weight_decay_lambda=0.036485686363384714 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Tunable_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.7948279106234177 --batch_size=126 --epochs=3 --lr=0.08877139494445535 --momentum=0.8566572279053658 --optimizer=sgd --weight_decay_lambda=0.8396867497885598 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done
