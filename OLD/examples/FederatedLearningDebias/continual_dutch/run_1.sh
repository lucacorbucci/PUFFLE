# Baseline
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Baseline_DP_2_Small_old_shift --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=66 --clipping=1.4261361024576042 --epochs=5 --lr=0.09671403740881956 --optimizer=adam --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --epsilon 2
done


# Fixed

for i in $(seq 20 22);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py  --run_name Fixed_DP_2_Small_old_shift --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=61 --clipping=4.07730642286986 --epochs=5 --lr=0.08667069976713908 --optimizer=adam --regularization_lambda=0.4487598162362314 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Fixed_DP_2_Small_old_shift --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=59 --clipping=17.887896797065956 --epochs=5 --lr=0.029727292546370524 --optimizer=sgd --regularization_lambda=0.1728379143354489 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 88 90);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Fixed_DP_2_Small_old_shift --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=52 --clipping=5.208187513746222 --epochs=3 --lr=0.06651275530529992 --optimizer=sgd --regularization_lambda=0.10072143651654804 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 100 102);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Fixed_DP_2_Small_old_shift --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=43 --clipping=8.298392769773965 --epochs=4 --lr=0.0220378754696751 --optimizer=sgd --regularization_lambda=0.04059977431368426 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done
