# celeba_local_tunable_disparity
for i in $(seq 0 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_reweight_celeba/../main.py  --run_name celeba_local_tunable_disparity --node_shuffle_seed $i --project_name reweighting_comparison_celeba  --alpha_target_lambda=2.831999533245229 --batch_size=478 --epochs=4 --lr=0.002203052632917548 --momentum=0.8823574512664099 --optimizer=adam --weight_decay_lambda=0.31415805486435894 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/ --seed 41  --wandb True --training_nodes 0.67 --test_nodes 0.335 --dataset_path ../../../../data/celeba/ --train_csv original_merged --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --approach representative --splitted_data_dir federated_4 --metric disparity --regularization_mode tunable --regularization True --target 0.1236 --comparison True
done

# celeba_local_fixed_disparity
for i in $(seq 0 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_reweight_celeba/../main.py  --run_name celeba_local_fixed_disparity --node_shuffle_seed $i --project_name reweighting_comparison_celeba   --batch_size=61 --epochs=4 --lr=0.08882938573472562 --optimizer=sgd --regularization_lambda=0.18088017224867664 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/ --train_csv original_merged --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --dataset_path ../../../../data/celeba/ --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --approach representative --splitted_data_dir federated_3 --metric disparity --regularization_mode fixed --regularization True --target 0.1236 --comparison True
done

# celeba_global_fixed_disparity
for i in $(seq 0 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_reweight_celeba/../main.py  --run_name celeba_global_fixed_disparity --node_shuffle_seed $i --project_name reweighting_comparison_celeba   --batch_size=49 --epochs=4 --lr=0.09741502978322708 --optimizer=sgd --regularization_lambda=0.4467682536501155 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/ --train_csv original_merged --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --dataset_path ../../../../data/celeba/ --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --partition_type representative --splitted_data_dir federated_3 --metric disparity --regularization_mode fixed --regularization True --target 0.1334 --comparison True
done

# celeba_global_tunable_disparity
for i in $(seq 0 5);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/comparison_with_reweight_celeba/../main.py  --run_name celeba_global_tunable_disparity --node_shuffle_seed $i --project_name reweighting_comparison_celeba  --alpha_target_lambda=1.797214685069447 --batch_size=38 --epochs=4 --lr=0.04850621041893968 --momentum=0.0646160587743968 --optimizer=sgd --weight_decay_lambda=0.007383539568968262 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.2 --sampled_clients_test 1 --debug True --base_path ../../../../data/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --dataset_path ../../../../data/celeba/ --train_csv original_merged --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --approach representative --splitted_data_dir federated_4 --metric disparity --regularization_mode tunable --regularization True --target 0.1334 --comparison True
done