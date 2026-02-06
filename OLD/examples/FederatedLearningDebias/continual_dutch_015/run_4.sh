



for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.623192519347804 --batch_size=96 --clipping=1.6892996673487377 --epochs=6 --lr=0.0663735568745178 --momentum=0.6538659277060359 --optimizer=adam --weight_decay_lambda=0.30599299549565406 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_015/../main.py --run_name 015_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.940330739055125 --batch_size=126 --clipping=1.6489322321513735 --epochs=6 --lr=0.06391821372930802 --momentum=0.3250051395342941 --optimizer=adam --weight_decay_lambda=0.17432418643143932 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_015/ --dataset_path ../../../../data/continual_dutch_015/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_4 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done





