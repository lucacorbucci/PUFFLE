
for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --batch_size=1559  --run_name Fixed_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i  --clipping=17.488122741197206 --epochs=5 --lr=0.020838816205347232 --optimizer=adam --regularization_lambda=0.2039969984493815 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --batch_size=1280 --run_name Fixed_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i --clipping=9.37381200690559 --epochs=4 --lr=0.0604308170535573 --optimizer=adam --regularization_lambda=0.39040798890439293 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --batch_size=2738 --run_name Fixed_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i --clipping=7.292046653626338 --epochs=3 --lr=0.02257391221709827 --optimizer=adam --regularization_lambda=0.08185657325897563 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --batch_size=3408 --run_name Fixed_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i --clipping=4.172985794202292 --epochs=5 --lr=0.06701744703750658 --optimizer=adam --regularization_lambda=0.5165274789892381 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode fixed --regularization True --target 0.1 --epsilon 1.5 --epsilon_statistics 0.5
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Tunable_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i --alpha_target_lambda=1.4263717836208472 --batch_size=1378 --clipping=2.6228817790021717 --epochs=2 --lr=0.06839396789277243 --momentum=0.801459647971662 --optimizer=adam --weight_decay_lambda=0.96104980126466 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Tunable_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i--alpha_target_lambda=1.29025308406372 --batch_size=3187 --clipping=3.4122354766639846 --epochs=3 --lr=0.019087088467296977 --momentum=0.3859730359716664 --optimizer=adam --weight_decay_lambda=0.19504793943777227 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Tunable_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i--alpha_target_lambda=3.4543753722419024 --batch_size=3641 --clipping=11.90775315152261 --epochs=3 --lr=0.04368038046921869 --momentum=0.7675748583323228 --optimizer=adam --weight_decay_lambda=0.4937261952622706 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Tunable_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i--alpha_target_lambda=3.2778506423840583 --batch_size=4117 --clipping=3.6000612913095567 --epochs=2 --lr=0.032387597198085406 --momentum=0.7667220816676357 --optimizer=adam --weight_decay_lambda=0.4863524569702626 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

for i in $(seq 4 6);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/FederatedLearningDebias/continual_income_2/../main.py --run_name Tunable_Income_2 --project_name Shift_test_new_data --node_shuffle_seed $i--alpha_target_lambda=2.288901998820522 --batch_size=2955 --clipping=18.205572735658603 --epochs=2 --lr=0.061682130693653706 --momentum=0.8133744751358055 --optimizer=adam --weight_decay_lambda=0.7278654127925324 --dataset continual_income --num_rounds 20 --num_client_cpus 1 --num_client_gpus 0.05 --pool_size 150 --sampled_clients 0.18 --sampled_clients_test 1 --debug False --base_path ../../../../data/continual_income_2/ --dataset_path ../../../../data/continual_income_2/ --seed 42 --wandb True --training_nodes 0.67 --test_nodes 0.335 --tabular_data True --metric disparity --splitted_data_dir federated --switch_dataset 11 --ratio_unfair_nodes 0.5 --regularization_mode tunable --regularization True --target 0.1 --epsilon 0.5 --epsilon_statistics 0.5 --epsilon_lambda 1 --global_computation True
done

