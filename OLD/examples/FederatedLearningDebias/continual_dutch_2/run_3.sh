

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 010_to_030_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.884047620392336 --batch_size=122 --clipping=2.7225010063504556 --epochs=5 --lr=0.06273673385711977 --momentum=0.5165274067420241 --optimizer=adam --weight_decay_lambda=0.21619388711117973 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 010_to_030_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.16871476775617672 --batch_size=50 --clipping=4.194417492176314 --epochs=2 --lr=0.09891680981708448 --momentum=0.4284487693803058 --optimizer=sgd --weight_decay_lambda=0.6796737473435267 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done
