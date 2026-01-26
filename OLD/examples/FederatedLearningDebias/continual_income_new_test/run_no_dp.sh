# Baseline_DP_1
for i in $(seq 2 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_new_test/../main.py --run_name Baseline_NO_DP --node_shuffle_seed $i --project_name Continual_Income_Two_switch --batch_size=2135 --epochs=5 --lr=0.0563771521064479 --optimizer=adam --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5
done
# 
# #  fixed_005_DP_2
for i in $(seq 2 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_new_test/../main.py --run_name Fixed_NO_DP --node_shuffle_seed $i --project_name Continual_Income_Two_switch --batch_size=1504 --epochs=5 --lr=0.08338669355004759 --optimizer=adam --regularization_lambda=0.4515161056937186 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done

# tunable_005_DP_2
for i in $(seq 2 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_new_test/../main.py --run_name Tunable_NO_DP --node_shuffle_seed $i --project_name Continual_Income_Two_switch --alpha_target_lambda=0.6374066119210444 --batch_size=2125 --epochs=3 --lr=0.03971820102248126 --momentum=0.8832303011485608 --optimizer=adam --weight_decay_lambda=0.6434760243214164 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done