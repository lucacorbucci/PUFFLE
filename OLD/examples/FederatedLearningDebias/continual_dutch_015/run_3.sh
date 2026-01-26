





for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.6829788215183112 --batch_size=113 --clipping=1.0964696275510692 --epochs=3 --lr=0.07836431515512464 --momentum=0.08500278299114293 --optimizer=adam --weight_decay_lambda=0.5912445558070037 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.8018913172777935 --batch_size=97 --clipping=1.293032056595141 --epochs=6 --lr=0.06939493045135185 --momentum=0.1328755914618209 --optimizer=adam --weight_decay_lambda=0.4554006091949564 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done


