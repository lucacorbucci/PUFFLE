# Private Baseline Dutch Cross-Device Epsilon 0.5
for i in $(seq 1 4);
do
    uv run ../../../../FlowerFLTemplate/main.py --batch_size=205 --lr=0.0847776665953605 --max_grad_norm=18.62380835595376 --num_epochs=3 --optimizer=adam --dataset_name dutch --num_client_cpus 1 --num_client_gpus 0 --num_rounds 10 --num_clients 150 --FL_setting cross_device --sampled_train_nodes_per_round 0.3 --sampled_validation_nodes_per_round 0 --sampled_test_nodes_per_round 1 --fed_dir ./training_data/ --project_name PuffleRefactoring --run_name Baseline_private_0.5 --wandb True --dataset_path ../../data/dutch_csv/dutch.csv --partitioner_type fairness --fairness_metric disparity --num_train_nodes 100 --num_validation_nodes 0 --num_test_nodes 50 --ratio_unfair_clients 0.5 --group_to_reduce 0 1 --group_to_increment 1 1 --ratio_unfairness 0.2 0.4 --target_attribute occupation --sensitive_attribute sex --epsilon 0.5 --distribution_mode representative
done

# Private Baseline Dutch Cross-Device Epsilon 1
for i in $(seq 1 4);
do
    uv run ../../../../FlowerFLTemplate/main.py --batch_size=70 --lr=0.09735784809384448 --max_grad_norm=3.557032802761245 --num_epochs=2 --optimizer=adam --dataset_name dutch --num_client_cpus 1 --num_client_gpus 0 --num_rounds 10 --num_clients 150 --FL_setting cross_device --sampled_train_nodes_per_round 0.3 --sampled_validation_nodes_per_round 0 --sampled_test_nodes_per_round 1 --fed_dir ./training_data_2/ --project_name PuffleRefactoring --run_name Baseline_private_1.0 --wandb True --dataset_path ../../data/dutch_csv/dutch.csv --partitioner_type fairness --fairness_metric disparity --num_train_nodes 100 --num_validation_nodes 0 --num_test_nodes 50 --ratio_unfair_clients 0.5 --group_to_reduce 0 1 --group_to_increment 1 1 --ratio_unfairness 0.2 0.4 --target_attribute occupation --sensitive_attribute sex --epsilon 1 --distribution_mode representative
done