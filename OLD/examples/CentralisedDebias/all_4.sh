


# celeba_007_NO_DP
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_007_NO_DP --project_name celeba_centralised_paper  --seed $i --batch_size=932 --epochs=10 --lr=0.0022427347899160784 --optimizer=adam --regularization_lambda=0.6586833133046055 --regularization_mode=fixed  --target=0.07
done

# celeba_011_NO_DP
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_011_NO_DP --project_name celeba_centralised_paper  --seed $i --batch_size=951 --epochs=10 --lr=0.08647050630251203 --optimizer=sgd --regularization_lambda=0.4315546704574473 --regularization_mode=fixed  --target=0.11
done
