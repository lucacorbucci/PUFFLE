poetry run python ./main.py --batch_size=278 --clipping=8.49474753594013 --epochs=5 --lr=0.0460886474538264 --node_shuffle_seed=265267533 --optimizer=adam --dataset dutch --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.3 --sampled_clients_test 1 --debug True --base_path ../data/Tabular --private True --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode no_tuning --tabular_data True --dataset_path ../data/Tabular/dutch/ --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --approach representative --epsilon 1 --project_name Dutch_test_federated --run_name Baseline_1


poetry run python ./main.py --alpha_target_lambda=3.645210016895725 --batch_size=232 --clipping=13.9079821394431 --epochs=5 --lr=0.07572970648406019 --momentum=0.50690219349591 --node_shuffle_seed=576878396 --optimizer=adam --weight_decay_lambda=0.9324356306287443 --dataset dutch --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.3 --sampled_clients_test 1 --debug True --base_path ../data/Tabular --DPL True --private True --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --starting_lambda_mode disparity --update_lambda True --tabular_data True --dataset_path ../data/Tabular/dutch/ --target 0.15 --group_to_reduce 0 1 --group_to_increment 1 1 --number_of_samples_per_node 343 --ratio_unfair_nodes 0.5 --ratio_unfairness 0 0 --approach representative --epsilon_lambda 0.25 --epsilon 0.5 --epsilon_statistics 0.25  --project_name Dutch_test_federated --run_name Dutch_01_epsilon_1_tunable