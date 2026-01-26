



for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.13856627100488603 --batch_size=121 --epochs=3 --lr=0.06418920353641305 --momentum=0.17860030237698263 --optimizer=adam --weight_decay_lambda=0.210984393626018 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_025/../main.py --run_name 025_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.9001800877168867 --batch_size=106 --epochs=5 --lr=0.03620769209749886 --momentum=0.37991152031710707 --optimizer=adam --weight_decay_lambda=0.2884444931339353 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_025/ --dataset_path ../../../../data/continual_dutch_025/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done
