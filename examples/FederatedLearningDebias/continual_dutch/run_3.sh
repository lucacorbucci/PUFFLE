
for i in $(seq 1 1);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Tunable_DP_2_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.5483772235828455 --batch_size=49 --clipping=4.0669793938045515 --epochs=7 --lr=0.09834566386684657 --momentum=0.18357339416944216 --optimizer=adam --weight_decay_lambda=0.14837840392375518 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.4 --epsilon_lambda 1.6
done

for i in $(seq 1 1);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Tunable_DP_2_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.0054232366864595 --batch_size=85 --clipping=1.3458018179515197 --epochs=5 --lr=0.06773979122026734 --momentum=0.33830101401069806 --optimizer=sgd --weight_decay_lambda=0.17854168908929066 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.4 --epsilon_lambda 1.6
done

for i in $(seq 1 1);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Tunable_DP_2_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.2525972653651439 --batch_size=122 --clipping=9.958110742812272 --epochs=5 --lr=0.05862773823133988 --momentum=0.1252685342229369 --optimizer=sgd --weight_decay_lambda=0.9904521369699524 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.4 --epsilon_lambda 1.6
done


for i in $(seq 1 1);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch/../main.py --run_name Tunable_DP_2_new --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.7379582424302322 --batch_size=86 --clipping=3.310942825788545 --epochs=4 --lr=0.053035037469102174 --momentum=0.5979831596404241 --optimizer=sgd --weight_decay_lambda=0.7506114797360531 --dataset continual_dutch --num_rounds 50 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch/ --dataset_path ../../../../data/continual_dutch/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.4 --epsilon_lambda 1.6
done
