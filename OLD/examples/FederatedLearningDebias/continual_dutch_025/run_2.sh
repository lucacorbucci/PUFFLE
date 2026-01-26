



for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.7868084587677159 --batch_size=111 --epochs=3 --lr=0.04620095193454995 --momentum=0.14172298830369978 --optimizer=adam --weight_decay_lambda=0.6142711765975298 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.9136571164334638 --batch_size=114 --epochs=2 --lr=0.0931042218233439 --momentum=0.6773055604699191 --optimizer=adam --weight_decay_lambda=0.41642873832974703 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

