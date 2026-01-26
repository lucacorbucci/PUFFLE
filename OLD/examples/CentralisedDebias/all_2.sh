# celeba_005_dp_5
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_005_dp_5 --project_name celeba_centralised_paper  --seed $i --batch_size=1962 --clipping_value=18.121892887830235 --epochs=10 --epsilon=5 --lr=0.021109166050500425 --optimizer=adam --regularization_lambda=0.6471705791374025 --regularization_mode=fixed  --target=0.05
done


# celeba_005_NO_DP
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_005_NO_DP --project_name celeba_centralised_paper  --seed $i --batch_size=1418 --epochs=10 --lr=0.04834985176849343 --optimizer=sgd --regularization_lambda=0.428263078319064 --regularization_mode=fixed  --target=0.05
done


# celeba_011_dp_5
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py  --run_name celeba_011_dp_5 --project_name celeba_centralised_paper --seed $i --batch_size=1639 --clipping_value=16.83435590038919 --epochs=10 --epsilon=5 --lr=0.05636358199881838 --optimizer=sgd --regularization_lambda=0.19974837424403225 --regularization_mode=fixed  --target=0.11
done


