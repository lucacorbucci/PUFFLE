# Baseline
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Baseline_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=40 --clipping=3.0513442009407608 --epochs=4 --lr=0.08198983591931977 --optimizer=sgd --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --epsilon 2
done

# Fixed
for i in $(seq 4 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Fixed_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=113 --clipping=2.1148250352767928 --epochs=4 --lr=0.0800835606527503 --optimizer=adam --regularization_lambda=0.4626218054200078 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 7 10);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Fixed_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=119 --clipping=1.9374091691868836 --epochs=2 --lr=0.0963261910851792 --optimizer=adam --regularization_lambda=0.28390980604799937 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 12 15);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Fixed_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=109 --clipping=1.1796849044515167 --epochs=4 --lr=0.09758398535463927 --optimizer=adam --regularization_lambda=0.015311723283305831 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 16 19);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Fixed_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=63 --clipping=1.3732894455040228 --epochs=5 --lr=0.04307151998557251 --optimizer=adam --regularization_lambda=0.12605808486131118 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 20 23);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Fixed_DP_2_old_grad --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=41 --clipping=4.810731810659428 --epochs=5 --lr=0.07687336226034802 --optimizer=sgd --regularization_lambda=0.12796318074299515 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

