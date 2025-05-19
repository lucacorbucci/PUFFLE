# Baseline
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py  --run_name Baseline_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=80 --clipping=1.06906825598064 --epochs=5 --lr=0.05979510041394186 --optimizer=adam --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --epsilon 2
done

# Tunable
for i in $(seq 11 14);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py  --run_name Tunable_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.481819736526069 --batch_size=57 --clipping=2.8357437063014523 --epochs=2 --lr=0.07055393028336451 --momentum=0.8973028172158469 --optimizer=sgd --weight_decay_lambda=0.42241213245663634 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 8 11);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Tunable_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.1631807026114767 --batch_size=75 --clipping=18.00942205191009 --epochs=3 --lr=0.030968271477388388 --momentum=0.5401078947803469 --optimizer=sgd --weight_decay_lambda=0.5514072753823659 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 111 114);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Tunable_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.2017087741600625 --batch_size=87 --clipping=1.7036026054219686 --epochs=5 --lr=0.06831853552149751 --momentum=0.002710937071266051 --optimizer=adam --weight_decay_lambda=0.4624174306178499 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 154 158);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Tunable_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.2828265738158433 --batch_size=79 --clipping=4.223310051284904 --epochs=4 --lr=0.08592931563701081 --momentum=0.7207341519701188 --optimizer=sgd --weight_decay_lambda=0.09394324719327556 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

# Fixed
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Fixed_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=62 --clipping=6.202188452517863 --epochs=3 --lr=0.07761673508448293 --optimizer=sgd --regularization_lambda=0.12682733698393886 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 5 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Fixed_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=36 --clipping=11.137652113114092 --epochs=3 --lr=0.058182543364662005 --optimizer=adam --regularization_lambda=0.09821141928967692 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 10 12);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Fixed_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=102 --clipping=7.273368406436773 --epochs=5 --lr=0.08160547941685795 --optimizer=sgd --regularization_lambda=0.25573718052006034 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 14 16);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_3/../main.py --run_name Fixed_DP_2_Medium_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=60 --clipping=4.697362534048802 --epochs=4 --lr=0.09926297612614468 --optimizer=adam --regularization_lambda=0.004289834470243725 --dataset continual_dutch --num_rounds 40  --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_3/ --dataset_path ../../../../data/continual_dutch_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done
