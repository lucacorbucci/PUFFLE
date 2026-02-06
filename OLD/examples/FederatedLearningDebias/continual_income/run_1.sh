

# # fixed_005
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_005_DP_1 --node_shuffle_seed $i --project_name Continual_Income --batch_size=3362 --clipping=6.109021763606809 --epochs=2 --lr=0.09220550632051534 --optimizer=adam --regularization_lambda=0.4630463810812238 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5 --epsilon 1 --regularization_mode fixed --regularization True --target 0.05
# done


# # fixed_01
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_01_DP_1 --node_shuffle_seed $i --project_name Continual_Income --batch_size=2977 --clipping=5.113006241157492 --epochs=4 --lr=0.031305326168884075 --optimizer=adam --regularization_lambda=0.35543382450502853 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5 --epsilon 1 --regularization_mode fixed --regularization True --target 0.1
# done


# # fixed_015
# for i in $(seq 0 7)f;
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name fixed_015_DP_1 --node_shuffle_seed $i --project_name Continual_Income --batch_size=2619 --clipping=19.159569134984817 --epochs=4 --lr=0.08614149369211312 --optimizer=sgd --regularization_lambda=0.10856599018065843 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5 --epsilon 1 --regularization_mode fixed --regularization True --target 0.15
# done



# # tunable_005
# for i in $(seq 0 7);
# do
#     poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name tunable_005_DP_1 --node_shuffle_seed $i --project_name Continual_Income --alpha_target_lambda=1.69107055824578 --batch_size=4016 --clipping=5.886007659878584 --epochs=2 --lr=0.08163563014672293 --momentum=0.3883711306786199 --optimizer=adam --weight_decay_lambda=0.5809769845282841 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5 --epsilon 0.9 --epsilon_lambda 0.1 --regularization_mode tunable --regularization True --target 0.05
# done

#  tunable_01_NO_DP
for i in $(seq 0 7);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income/../main.py --run_name tunable_01_NO_DP --node_shuffle_seed $i --project_name Continual_Income --alpha_target_lambda=2.4687720058783733 --batch_size=2739 --epochs=5 --lr=0.007528797630143794 --momentum=0.48235146031023896 --optimizer=adam --weight_decay_lambda=0.6585303139262145 --dataset continual_income --num_rounds 45 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --sampled_clients_validation 0 --debug False --base_path ../../../../data/continual_income/ --dataset_path ../../../../data/continual_income/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 5 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1
done
