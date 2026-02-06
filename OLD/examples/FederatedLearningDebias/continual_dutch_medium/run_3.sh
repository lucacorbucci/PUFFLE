



for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.4519547957955551 --batch_size=119 --clipping=1.059280487905051 --epochs=4 --lr=0.07559833430980133 --momentum=0.5739324193061028 --optimizer=adam --weight_decay_lambda=0.6387618100434487 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.1908306836312215 --batch_size=53 --clipping=1.2622941512324677 --epochs=2 --lr=0.08686583138512806 --momentum=0.4131873996228819 --optimizer=adam --weight_decay_lambda=0.6455880781267571 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.26468213429259196 --batch_size=89 --clipping=2.886540267122216 --epochs=3 --lr=0.09024866045734076 --momentum=0.8762385837245122 --optimizer=adam --weight_decay_lambda=0.3927490924355097 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.8028791362188205 --batch_size=84 --clipping=1.4955935624276546 --epochs=5 --lr=0.06606585700320869 --momentum=0.4276778057929842 --optimizer=adam --weight_decay_lambda=0.3861204109269423 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.778252932540821 --batch_size=127 --clipping=1.558310654756772 --epochs=6 --lr=0.07388878606521432 --momentum=0.3231678068708859 --optimizer=adam --weight_decay_lambda=0.35304961234902243 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.1632881859948104 --batch_size=90 --clipping=1.2700685949319448 --epochs=7 --lr=0.08244673443669216 --momentum=0.45367770386540707 --optimizer=adam --weight_decay_lambda=0.6412642736562019 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.4670003585688858 --batch_size=68 --clipping=1.6015916608086784 --epochs=2 --lr=0.05616538785066027 --momentum=0.35207752677369636 --optimizer=adam --weight_decay_lambda=0.3482477572025702 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.28325043473800304 --batch_size=61 --clipping=3.1907641920648837 --epochs=3 --lr=0.06430972502008865 --momentum=0.6094936416155796 --optimizer=adam --weight_decay_lambda=0.05244824832121575 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

