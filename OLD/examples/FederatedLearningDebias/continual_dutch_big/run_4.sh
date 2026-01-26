
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Tunable_NO_privacy_4 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.183477544502462 --batch_size=113 --epochs=1 --lr=0.07073077232814391 --momentum=0.2852039278356135 --optimizer=adam --weight_decay_lambda=0.3099554711557848 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Tunable_NO_privacy_4 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.2773994219054718 --batch_size=107 --epochs=2 --lr=0.04349611199343004 --momentum=0.8509813192585295 --optimizer=adam --weight_decay_lambda=0.17886539498816828 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Tunable_NO_privacy_4 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.2773994219054718 --batch_size=107 --epochs=2 --lr=0.04349611199343004 --momentum=0.8509813192585295 --optimizer=adam --weight_decay_lambda=0.17886539498816828 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Tunable_NO_privacy_4 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.8299988747520445 --batch_size=102 --epochs=7 --lr=0.08478781292418543 --momentum=0.7458997988681904 --optimizer=sgd --weight_decay_lambda=0.5854032387502064 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_big/../main.py --run_name Tunable_NO_privacy_4 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.19799488203176657 --batch_size=118 --epochs=1 --lr=0.08858380958208956 --momentum=0.7544650050754386 --optimizer=adam --weight_decay_lambda=0.9760401834661212 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.10 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_big/ --dataset_path ../../../../data/continual_dutch_big/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done
