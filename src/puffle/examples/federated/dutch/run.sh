# Baseline Dutch Cross-device
for i in $(seq 1 4);
do
    uv run  ../../../../FlowerFLTemplate/main.py --project_name PuffleRefactoring --run_name Dutch_Baseline --node_shuffle_seed $i --batch_size=67 --lr=0.05344451619469467 --num_epochs=3 --optimizer=sgd --dataset_name dutch --num_client_cpus 1 --num_client_gpus 0 --num_rounds 10 --num_clients 150 --FL_setting cross_device --sampled_train_nodes_per_round 0.3 --sampled_validation_nodes_per_round 0 --sampled_test_nodes_per_round 1 --fed_dir ./training_data/ --wandb True --dataset_path ../../data/dutch_csv/dutch.csv --partitioner_type fairness --fairness_metric disparity --num_train_nodes 100 --num_validation_nodes 0 --num_test_nodes 50 --ratio_unfair_clients 0.5 --group_to_reduce 1 0 --group_to_increment 1 1 --ratio_unfairness 0.5 0.6 --target_attribute occupation --sensitive_attribute sex
done

# Private Baseline Dutch Cross-Device
for i in $(seq 1 4);
do
    uv run ../../../../FlowerFLTemplate/main.py --project_name PuffleRefactoring --run_name Dutch_Private_Baseline --node_shuffle_seed $i --batch_size=228 --lr=0.048780294806384634 --max_grad_norm=18.743904033810526 --num_epochs=3 --optimizer=adam --dataset_name dutch --num_client_cpus 1 --num_client_gpus 0 --num_rounds 10 --num_clients 150 --FL_setting cross_device --sampled_train_nodes_per_round 0.3 --sampled_validation_nodes_per_round 0 --sampled_test_nodes_per_round 1 --fed_dir ./training_data/ --wandb True --dataset_path ../../data/dutch_csv/dutch.csv --partitioner_type fairness --fairness_metric disparity --num_train_nodes 100 --num_validation_nodes 0 --num_test_nodes 50 --ratio_unfair_clients 0.5 --group_to_reduce 1 0 --group_to_increment 1 1 --ratio_unfairness 0.5 0.6 --target_attribute occupation --sensitive_attribute sex --epsilon 2
done