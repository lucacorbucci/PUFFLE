







for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.5843594473443231 --batch_size=72 --clipping=2.2020378868259907 --epochs=3 --lr=0.05991172431259475 --momentum=0.499947523448554 --optimizer=adam --weight_decay_lambda=0.23410312611604167 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.7887141470619113 --batch_size=103 --clipping=2.1997273524057395 --epochs=5 --lr=0.09879839793170105 --momentum=0.731447124212287 --optimizer=adam --weight_decay_lambda=0.18344219431702777 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done
