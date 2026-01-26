for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Baseline_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i  --batch_size=36 --clipping=2.7566958514837228 --epochs=7 --lr=0.07904571715271294 --optimizer=sgd --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --epsilon 2
done

# Tunable
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i   --alpha_target_lambda=3.575888715159805 --batch_size=114 --clipping=3.046166712809598 --epochs=4 --lr=0.08488224020693251 --momentum=0.04584835825504733 --optimizer=sgd --weight_decay_lambda=0.01955833751998447 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=0.5376081341991216 --batch_size=113 --clipping=5.053501064668858 --epochs=7 --lr=0.07219442143794576 --momentum=0.02336440709648261 --optimizer=sgd --weight_decay_lambda=0.92800711845036 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=1.460555983262671 --batch_size=56 --clipping=11.614867644302327 --epochs=5 --lr=0.046847189067969225 --momentum=0.7990454134080157 --optimizer=adam --weight_decay_lambda=0.6061030446536144 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=2.006090414565846 --batch_size=102 --clipping=2.364321694687888 --epochs=5 --lr=0.06481278235210713 --momentum=0.21279257111550576 --optimizer=sgd --weight_decay_lambda=0.34846039375794813 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Tunable_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --alpha_target_lambda=3.5615830695479738 --batch_size=53 --clipping=13.09195088187284 --epochs=6 --lr=0.03829833479704797 --momentum=0.5222738918209058 --optimizer=sgd --weight_decay_lambda=0.38288917629760016 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.05 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

# Fixed 
for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Fixed_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=97 --clipping=10.169285536539109 --epochs=4 --lr=0.03153503718041769 --optimizer=sgd --regularization_lambda=0.21841528017225845 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Fixed_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=87 --clipping=12.526723127919253 --epochs=5 --lr=0.04413935276164466 --optimizer=sgd --regularization_lambda=0.22552608772019 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Fixed_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=53 --clipping=2.5357675449596635 --epochs=2 --lr=0.08442851573328106 --optimizer=sgd --regularization_lambda=0.47168317520722913 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 1 3);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_dutch_inverse_2/../main.py --run_name Fixed_inverse_old_grad_DP_2 --project_name Shift_Comparison --node_shuffle_seed $i --batch_size=90 --clipping=5.860197787467724 --epochs=5 --lr=0.025389881864048152 --optimizer=sgd --regularization_lambda=0.0790549490556396 --dataset continual_dutch --num_rounds 40 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 11 --debug False --base_path ../../../../data/continual_dutch_inverse_2/ --dataset_path ../../../../data/continual_dutch_inverse_2/ --seed 42 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 21 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.05 --epsilon 1.5 --epsilon_statistics 0.5
done

