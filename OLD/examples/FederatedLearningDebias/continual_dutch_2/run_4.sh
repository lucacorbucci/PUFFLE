
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 010_to_030_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.5893549830076927 --batch_size=104 --clipping=1.44411467631228 --epochs=4 --lr=0.0935799979494148 --momentum=0.6643668843674657 --optimizer=adam --weight_decay_lambda=0.5566429844766769 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 010_to_030_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.1414000245024762 --batch_size=101 --clipping=1.7273944227528957 --epochs=2 --lr=0.08846915696673702 --momentum=0.2152775570777393 --optimizer=adam --weight_decay_lambda=0.7311969340408331 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 010_to_030_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.3758350455010535 --batch_size=127 --clipping=1.8204117533208088 --epochs=6 --lr=0.0754377321907637 --momentum=0.612173053371202 --optimizer=adam --weight_decay_lambda=0.29520141171612835 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done
