# PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning 
    
[![arXiv](https://img.shields.io/badge/arXiv-2407.15224-b31b1b.svg)](https://arxiv.org/abs/2407.15224)
[![ECAI](https://img.shields.io/badge/ECAI-2024-blue.svg)](https://www.ecai2024.eu/)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

This repository contains the implementation of the paper "PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning" by L. Corbucci, M. A. Heikkilä, D.S. Noguero, A. Monreale, N. Kourtellis. The paper has been accepted at the European Conference of Artificial Intelligence (ECAI) 2024. The preprint of the paper can be found [**here**](https://arxiv.org/abs/2407.15224).

**Abstract**: Training and deploying Machine Learning models that simultaneously adhere to principles of fairness and privacy while ensuring good utility poses a significant challenge. The interplay between these three factors of trustworthiness is frequently underestimated and remains insufficiently explored. Consequently, many efforts focus on ensuring only two of these factors, neglecting one in the process. The decentralization of the datasets and the variations in distributions among the clients exacerbate the complexity of achieving this ethical trade-off in the context of Federated Learning (FL). For the first time in FL literature, we address these three factors of trustworthiness. We introduce PUFFLE, a high-level parameterised approach that can help in the exploration of the balance between utility, privacy, and fairness in FL scenarios. We prove that PUFFLE can be effective across diverse datasets, models, and data distributions, reducing the model unfairness up to 75%, with a maximum reduction in the utility of 17% in the worst-case scenario, while maintaining strict privacy guarantees during the FL training.

![PUFFLE Logo - generated with Adobe Firefly](/puffle_logo.png)

## How to install the dependencies

I used Poetry as dependency manager. If you don't have poetry installed you can run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, we can install all the dependencies:

- poetry install 

## Datasets

In the paper we used three datasets to evaluate the performance of PUFFLE: Dutch, Celeba and ACS Income.

- The Dutch Dataset can be found [here](https://raw.githubusercontent.com/tailequy/fairness_dataset/main/Dutch_census/dutch_census_2001.arff). Once you download it, it is enough to put into the folder /data/Tabular/dutch (create the folder before if it does not exist). If you prefer to put it into a different folder, you can do it. Remember that you'll have to change the path in the configurations file to run the hyperparameter tuning. You also have to create the folder /data/Tabular/dutch/federated to store the splitted data.
- The Celeba dataset can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), we used the img_align_dataset. Once you download it, it is enough to put into the folder /data/celeba (creaate the folder before if it does not exist). If you prefer to put it into a different folder, you can do it. Remember that you'll have to change the path in the configurations file to run the hyperparameter tuning. For this dataset, we need to use a csv file with the metadata of the images. You can find the csv file in [experiments/data_for_celeba/](/experiments/data_for_celeba/original_merged.csv). You also have to create the folder /data/celeba/celeba-10-batches-py/federated to store the splitted data.
- For the Income Dataset, please refer to the [paper](https://arxiv.org/abs/2108.04884).

### Add your custom dataset

To use PUFFLE with another dataset, you need to create a new class like [the one we have for Celeba](/puffle/Utils/celeba.py) or the [TabularDataset](/puffle/Utils/dutch.py) used for Dutch. 
In particular, for tabular datasets, you could even use the already implemented [TabularDataset](/puffle/Utils/dutch.py).
It is important to note that the dataset should have three "information": the actual training data, a target and a sensitive group. The sensitive group is used to calculate the fairness metrics.
This is why in the __get__item() method of the dataset class, we return three values. 

Once you have defined the dataset class, you need to add it to the get_model() method in the [model_utils.py](/puffle/Utils/model_utils.py) where you define the corresponding model class to use for the dataset. If you need a transformation function, you need to add an option for your dataset in the get_transformation() method in [utils.py](/puffle/Utils/utils.py). Lastly, you have to add your dataset to the [dataset_utils.py](/puffle/Utils/dataset_utils.py) file in order to load the dataset from disk.

### Add your custom model

To use PUFFLE with another model, you need to create a new class like [the one we have for Celeba](/puffle/Models/celeba_net.py). Then, you have to change the get_model() method in the [model_utils.py](/puffle/Utils/model_utils.py) to return the new model class. Note that some model architecture could not work with the current implementation of PUFFLE. In particular, the use of Opacus put some constraints on the model architecture, for more information please refer to the [Opacus documentation](https://opacus.ai/api/validator.html).

## How to run the experiments

We rely on Wandb to log the experiments and to perform the hyperparameter search. You need to create an account in Wandb and to login in your terminal to be able to run the experiments. For more information on how to install and login in Wandb, please refer to the [**official documentation**](https://docs.wandb.ai/quickstart).

In the paper we used three datasets to run the experiments, in the /experiments folder, you'll find the configurations files to run the sweep for each dataset. For each dataset you will have a configuration file for the baseline, two configuration files for the baseline with Differential Privacy, one for each privacy budget, and several configuration files for PUFFLE. The different PUFFLE configurations file are used to consider multiple privacy budgets and fairness requirements.
For the settings that you can change in the configuration file, please refer to the [main.py](/puffle/main.py) file where we explained the meaning of the parameters.

Inside each folder, you can also find a run_sweep_1.sh file that you can use to run the hyperparameter search. You can run the experiments by running the following command:

```
sh run_sweep_1.sh
```

An example of the hyperparameter search for the Celeba Dataset can be found [here](https://wandb.ai/lucacorbucci/PUFFLE_Celeba/sweeps). In the sweep tabs you can find the sweep that we ran for the Baseline and the sweep that we ran for PUFFLE with Tunable Lambda with Epsilon=5 and Target Unfairness set to 0.06

## An example of how to run the experiments:

It is also possible to run a single experiment without the hyperparameter search. We did this to run the best configuration found in the hyperparameter search with multiple seeds to validate the results.

For instance, if you want to run a simple experiment with Celeba Dataset and an IID data distribution, you can run the following command (you can change the parameters as you prefer, I have not done the hyp search for this example, it is just to show how to run a single experiment in this case):

```
poetry run python /usr/bin/env poetry run python ../../puffle/main.py --node_shuffle_seed 111 --batch_size=96 --epochs=4 --lr=0.03307371632575612 --optimizer=sgd --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.34 --sampled_clients_test 1.0 --train_csv original_merged --debug False --base_path ../../data/ --dataset_path ../../data/celeba/ --seed 41 --wandb True  --training_nodes 0.67 --test_nodes 0.335 --partition_type iid --splitted_data_dir federated_3
```

To run a baseline Celeba experiment with a non-iid data distribution, you can run the following command (this one is the best configuration found in the hyperparameter search with the node_shuffle_seed equal to 111 to change the order in which the training nodes are selected. The test set is always the same during the experiments):

```
poetry run python ../../puffle/main.py --node_shuffle_seed 111 --batch_size=200 --clipping=8.39716744180798 --epochs=4 --lr=0.07739668230018099 --optimizer=sgd --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.34 --sampled_clients_test 1.0 --train_csv original_merged --debug False --base_path ../../data/ --dataset_path ../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated --epsilon 5
```

Lastly, to run a PUFFLE experiment with Celeba Dataset, a non-IID data distribution, a Tunable Lambda, epsilon=5 and fairness target equal to 0.06 you can run the following command (this one is the best configuration found in the hyperparameter search with the node_shuffle_seed equal to 111 to change the order in which the training nodes are selected. The test set is always the same during the experiments):

```
poetry run python ../../puffle/main.py --node_shuffle_seed 111 --alpha_target_lambda=0.15763682117751202 --batch_size=397 --clipping=4.748517653719572 --epochs=3 --lr=0.08223497382436028 --momentum=0.3248362103462307 --optimizer=sgd --weight_decay_lambda=0.647343990184165 --dataset celeba --num_rounds 39 --num_client_cpus 1 --num_client_gpus 0.1 --pool_size 150 --sampled_clients 0.34 --sampled_clients_test 1.0 --train_csv original_merged --debug False --base_path ../../data/ --dataset_path ../../data/celeba/ --seed 41 --wandb True --training_nodes 0.67 --test_nodes 0.335 --partition_type representative --group_to_reduce 1 1 --group_to_increment 0 1 --number_of_samples_per_node 1350 --ratio_unfair_nodes 0.5 --ratio_unfairness 0.9 0.9 --one_group_nodes True --splitted_data_dir federated --metric disparity --epsilon 4 --epsilon_statistics 0.5 --epsilon_lambda 0.5 --update_lambda True --regularization_mode tunable --regularization True --target 0.06
```

For the last command, you can see [here](https://wandb.ai/lucacorbucci/PUFFLE_Celeba/runs/sbir9cxu?nw=nwuserlucacorbucci) the result of the experiment that we ran after the hyperparameter search putting a node_shuffle_seed equal to 111. If you search for "Test Disparity with statistics" you can see the disparity of the model on the test set. If you search for "Test Accuracy
" you can see the accuracy of the model on the test set. 


## How to cite this work

If you use this code, please cite the following paper:

```
@misc{corbucci2024pufflebalancingprivacyutility,
      title={PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning}, 
      author={Luca Corbucci and Mikko A Heikkila and David Solans Noguero and Anna Monreale and Nicolas Kourtellis},
      year={2024},
      eprint={2407.15224},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15224}, 
}
```