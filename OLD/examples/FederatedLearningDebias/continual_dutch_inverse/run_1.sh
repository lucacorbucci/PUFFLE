# for i in $(seq 4 6);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Tunable_new_alg --project_name Shift_Comparison --node_shuffle_seed $i  --alpha_target_lambda=2.956451301039951 --batch_size=127 --clipping=12.9174114642235 --epochs=3 --lr=0.061194578138545105 --momentum=0.16576137136251795 --optimizer=sgd --weight_decay_lambda=0.5311637864212553 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 
# done

# for i in $(seq 4 6);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Tunable_new_alg --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.4971899706651386 --batch_size=47 --clipping=3.5557753328882375 --epochs=6 --lr=0.09971708425371496 --momentum=0.25024546906601935 --optimizer=sgd --weight_decay_lambda=0.14075984477666717 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2
# done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Fixed_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=42 --epochs=5 --lr=0.0734675699966163 --optimizer=sgd --regularization_lambda=0.1784217694106301 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Fixed_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=50 --epochs=5 --lr=0.014353826703973728 --optimizer=adam --regularization_lambda=0.5379549569437837 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Fixed_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=76 --epochs=6 --lr=0.08135507618778128 --optimizer=adam --regularization_lambda=0.4227042873239675 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse/../main.py --run_name Inverse_Fixed_NO_DP_new --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=55 --epochs=7 --lr=0.0981661788075776 --optimizer=sgd --regularization_lambda=0.00965072345070894 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_inverse/ --dataset_path ../../../../data/continual_dutch_inverse/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05
done

