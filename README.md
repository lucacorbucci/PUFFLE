# PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning 

This repository contains the implementation of the paper "PUFFLE: Balancing Privacy, Utility, and Fairness in Federated Learning" accepted at the European Conference of Artificial Intelligence (ECAI) 2024. The preprint of the paper can be found [**here**](https://arxiv.org/abs/2407.15224).

**Abstract**: Training and deploying Machine Learning models that simultaneously adhere to principles of fairness and privacy while ensuring good utility poses a significant challenge. The interplay between these three factors of trustworthiness is frequently underestimated and remains insufficiently explored. Consequently, many efforts focus on ensuring only two of these factors, neglecting one in the process. The decentralization of the datasets and the variations in distributions among the clients exacerbate the complexity of achieving this ethical trade-off in the context of Federated Learning (FL). For the first time in FL literature, we address these three factors of trustworthiness. We introduce PUFFLE, a high-level parameterised approach that can help in the exploration of the balance between utility, privacy, and fairness in FL scenarios. We prove that PUFFLE can be effective across diverse datasets, models, and data distributions, reducing the model unfairness up to 75%, with a maximum reduction in the utility of 17% in the worst-case scenario, while maintaining strict privacy guarantees during the FL training.

## How to install the dependencies

I used Poetry as dependency manager. If you don't have poetry installed you can run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, we can install all the dependencies:

- poetry install 

## Datasets

In the paper we used three datasets to evaluate the performance of PUFFLE: Dutch, Celeba and ACS Income.

- The Dutch Dataset can be found [here](https://raw.githubusercontent.com/tailequy/fairness_dataset/main/Dutch_census/dutch_census_2001.arff). Once you download it, it is enough to put into the folder /data/Tabular/dutch (creaate the folder before if it does not exist). If you prefer to put it into a different folder, you can do it. Remember that you'll have to change the path in the configurations file to run the hyperparameter tuning. 
- The Celeba dataset can be found [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), we used the img_align_dataset. Once you download it, it is enough to put into the folder /data/celeba (creaate the folder before if it does not exist). If you prefer to put it into a different folder, you can do it. Remember that you'll have to change the path in the configurations file to run the hyperparameter tuning.
- For the Income Dataset, please refer to the [paper](https://arxiv.org/abs/2108.04884).


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

For instance, if you want to run a simple experiment with Celeba Dataset and an IID data distribution, you can run the following command:

"""
"""

To run a baseline Celeba experiment with a non-iid data distribution, you can run the following command:

"""
"""

Lastly, to run a PUFFLE experiment with Celeba Dataset, a non-IID data distribution, a Tunable Lambda, epsilon=5 and fairness target equal to 0.06 you can run the following command:

"""
"""


# How to cite this work

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