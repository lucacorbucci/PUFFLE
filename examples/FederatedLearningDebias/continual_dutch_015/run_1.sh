




for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.1767268564913964 --batch_size=109 --clipping=1.1667018714454187 --epochs=7 --lr=0.06345526746078857 --momentum=0.43902153578741576 --optimizer=adam --weight_decay_lambda=0.06553879099745286 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.116096380602034 --batch_size=122 --clipping=1.138851939649209 --epochs=3 --lr=0.09193616993297944 --momentum=0.18640849276114663 --optimizer=adam --weight_decay_lambda=0.34297819573906224 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.3751173088870138 --batch_size=126 --clipping=1.9288212595406995 --epochs=5 --lr=0.08483952072824076 --momentum=0.5904736610480336 --optimizer=adam --weight_decay_lambda=0.29297535527936713 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done