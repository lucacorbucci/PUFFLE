for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i  --alpha_target_lambda=1.3197344159634137 --batch_size=76 --epochs=5 --lr=0.0466721049198753 --momentum=0.3653776286565731 --optimizer=adam --weight_decay_lambda=0.5326280917841939 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.1878345514793461 --batch_size=109 --epochs=4 --lr=0.08016837842026793 --momentum=0.22941215365005452 --optimizer=adam --weight_decay_lambda=0.260711565094552 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.9187427691333536 --batch_size=35 --epochs=2 --lr=0.09523903313704916 --momentum=0.16978954432919724 --optimizer=adam --weight_decay_lambda=0.1308426379016411 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.257299348270817 --batch_size=95 --epochs=2 --lr=0.08745068342350519 --momentum=0.18878748556167024 --optimizer=adam --weight_decay_lambda=0.33573000314804013 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.635304424743519 --batch_size=52 --epochs=6 --lr=0.01840398111668617 --momentum=0.11756550235346494 --optimizer=adam --weight_decay_lambda=0.1451009644555577 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.952938096357804 --batch_size=90 --epochs=5 --lr=0.033567213434585984 --momentum=0.05428999147870089 --optimizer=adam --weight_decay_lambda=0.3729518565452965 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done



