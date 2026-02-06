
# Tunable

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Baseline_Inverse_Tunable_Mid_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --run_name Tunable_Inverse_Tunable_Mid_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.168007355634544 --batch_size=43 --epochs=4 --lr=0.03977692244262139 --momentum=0.35585073837448167 --optimizer=adam --weight_decay_lambda=0.4549483860747988 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_Inverse_Tunable_Mid_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.7840549411839164 --batch_size=60 --epochs=2 --lr=0.019443260059455948 --momentum=0.7138884660430213 --optimizer=adam --weight_decay_lambda=0.8573377335101403 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done
