# baseline 
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py  --run_name baseline --project_name celeba_centralised_paper  --seed $i --batch_size=1757 --epochs=10 --lr=0.006546994739548253 --optimizer=adam 
done

# baseline_DP_5
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py  --run_name baseline_DP_5 --project_name celeba_centralised_paper --seed $i --batch_size=972 --clipping_value=6.07372665598714 --epochs=10 --epsilon=5 --lr=0.013458578485695576 --optimizer=adam 
done


# celeba_007_dp_5
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py  --run_name celeba_007_dp_5 --project_name celeba_centralised_paper --seed $i --batch_size=620 --clipping_value=5.52606129130794 --epochs=10 --epsilon=5 --lr=0.005003915041031188 --optimizer=adam --regularization_lambda=0.6630146254359192 --regularization_mode=fixed  --target=0.07
done

