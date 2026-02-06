


for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.951900357322223 --batch_size=119 --epochs=3 --lr=0.05754342449267903 --momentum=0.13317047125972237 --optimizer=adam --weight_decay_lambda=0.6644729515005823 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.29949796541813856 --batch_size=108 --epochs=5 --lr=0.054596324222737094 --momentum=0.19373424162057487 --optimizer=adam --weight_decay_lambda=0.6384450393132832 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_3 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done


