# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_1/../main.py --run_name Baseline_Income_1 --project_name Shift_test_new_data --node_shuffle_seed $i --batch_size=1148 --clipping=2.4392418006300995 --epochs=5 --lr=0.0665204836833222 --optimizer=adam --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_1/ --dataset_path ../../../../data/continual_income_1/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --epsilon 2
# done

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Baseline_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i --batch_size=2533 --clipping=6.3589146542836135 --epochs=5 --lr=0.09628733585465926 --optimizer=adam --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --epsilon 2
# done

# for i in $(seq 1 3);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_3/../main.py --run_name Baseline_Income_3 --project_name Shift_test_new_data --node_shuffle_seed $i --batch_size=4638 --clipping=2.870657096650361 --epochs=5 --lr=0.0994421901404623 --optimizer=adam --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_3/ --dataset_path ../../../../data/continual_income_3/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --epsilon 2
# done


# for i in $(seq 1 5);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_1/../main.py --run_name Fixed_Income_1_target_015 --project_name Shift_test_new_data --node_shuffle_seed $i  --batch_size=4994 --clipping=2.710458720752723 --epochs=5 --lr=0.0848553611543431 --optimizer=adam --regularization_lambda=0.12932269945624797 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_1/ --dataset_path ../../../../data/continual_income_1/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.15 --epsilon 1.5 --epsilon_statistics 0.5
# done

for i in $(seq 4 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_1/../main.py --run_name Tunable_Income_1_target_005_NO_DP_new --project_name Shift_test_new_data --node_shuffle_seed $i --alpha_target_lambda=2.256686084667922 --batch_size=1659 --epochs=2 --lr=0.05469131320780834 --momentum=0.7966223288147959 --optimizer=adam --weight_decay_lambda=0.7805897795567285 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.3 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_1/ --dataset_path ../../../../data/continual_income_1/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --global_computation True
done 

