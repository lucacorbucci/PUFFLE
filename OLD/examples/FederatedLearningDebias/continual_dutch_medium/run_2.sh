


for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.31228978398368523 --batch_size=106 --epochs=6 --lr=0.08341146892829238 --momentum=0.7169904726904761 --optimizer=adam --weight_decay_lambda=0.5544090982177237 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.2974871598018067 --batch_size=102 --epochs=3 --lr=0.04700833568542772 --momentum=0.24289526905491615 --optimizer=adam --weight_decay_lambda=0.3960115853569528 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.4480888295303416 --batch_size=104 --epochs=7 --lr=0.039061940589472544 --momentum=0.7374215568139928 --optimizer=adam --weight_decay_lambda=0.394510588262011 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_NO_DP --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.936962506200701 --batch_size=113 --epochs=5 --lr=0.029286939389474576 --momentum=0.12732120072268296 --optimizer=adam --weight_decay_lambda=0.22772621272970628 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05
done



for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.5100025414527118 --batch_size=122 --clipping=1.7526748865485005 --epochs=5 --lr=0.08429267532836517 --momentum=0.3801456381171264 --optimizer=adam --weight_decay_lambda=0.6122587443709612 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_medium/../main.py --run_name Medium_DP_5 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.3087589339641186 --batch_size=125 --clipping=1.1234742160625326 --epochs=7 --lr=0.08563177309706502 --momentum=0.6885597089019833 --optimizer=adam --weight_decay_lambda=0.48592529406301127 --dataset continual_dutch --num_rounds 60 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_dutch_medium/ --dataset_path ../../../../data/continual_dutch_medium/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated_2 --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 2 --epsilon_lambda 3
done




