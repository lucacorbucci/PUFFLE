# celeba_011_dp_5_tunable
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_011_dp_5_tunable --project_name celeba_centralised_paper --seed $i --alpha_target_lambda=2.960715639786792 --batch_size=1617 --clipping_value=11.405731318148703 --epochs=10 --epsilon=5 --lr=0.06348227538360061 --momentum=0.7613352968508602 --optimizer=sgd --regularization_mode=tunable  --target=0.11 --weight_decay_lambda=0.9099287776833224
done

# celeba_005_dp_5_tunable
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_005_dp_5_tunable --project_name celeba_centralised_paper --seed $i --alpha_target_lambda=0.3644572555817446 --batch_size=1595 --clipping_value=19.955682987328657 --epochs=10 --epsilon=5 --lr=0.08094765552201862 --momentum=0.06314277649012459 --optimizer=sgd --regularization_mode=tunable  --target=0.05 --weight_decay_lambda=0.9679573657574672
done


# celeba_007_dp_5_tunable
for i in $(seq 0 4);
do
    poetry run python /home/lcorbucci/Unfairness-Regularization/examples/CentralisedDebias/test_celeba.py --run_name celeba_007_dp_5_tunable --project_name celeba_centralised_paper --seed $i --alpha_target_lambda=0.3644572555817446 --batch_size=1595 --clipping_value=19.955682987328657 --epochs=10 --epsilon=5 --lr=0.08094765552201862 --momentum=0.06314277649012459 --optimizer=sgd --regularization_mode=tunable  --target=0.07 --weight_decay_lambda=0.9679573657574672
done
