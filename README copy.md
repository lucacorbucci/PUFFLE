# Training ML models with Regularization to reduce unfairness

This repository contains the implementation of an approach to reduce the unfairness of ML model during the training through the use of a regularization term. 
The regularization term depends on the unfairness level of the model during the training. 
At the moment we only implemented the Demographic Parity as unfairness metric, as explained in the paper [Learning with Impartiality to Walk on the Pareto Frontier of Fairness, Privacy, and Utility](). However, we plan to introduce similar metrics in the future.

# Structure of the repository

The repository is structured as follows, the DPL folder contains the main code of the library. This folder contains three subfolders:
- DPLUtils: some utility functions to compute the fairness metrics and to manipulate the data.
- Learning: the implementation of the training and test method to train the model using the regularization term.
- Regularization term: the implementation of the computation of the regularization term.

# How to run the code

## Requirements

- I used Poetry as dependency manager. If you have poetry installed, you can skip this step. If you don't have poetry installed, you can find the instructions to install it [here](https://python-poetry.org/docs/).
- Once you have poetry installed, you can run:

```
poetry install 
```

to create a virtualenv and install all the dependencies. 

## How to use the library


Since we need to use this approach with Differential Privacy (DP), we need to train a "private" model using the Opacus library. However, it is possible to run the experiment without DP just by setting a couple of parameters. 

To use the library we need to create a RegularizationConfig, a configuration object that contains our preferences. For instance, with the following code we can create a RegularizationConfig object to train a model. Inside this object you can set the following parameters:
- epochs: the number of epochs to train the model
- device: the device to use for the training
- batch_size: the batch size to use for the training
- seed: the seed to use for the training
- regularization: a boolean to indicate if we want to use the regularization term or not. 
- target: the target value for the unfairness metric. 
- regularization_mode: the type of regularization we want to use. At the moment we only have "tunable" and "fixed". With tunable we update the value of the Lambda during the training, with fixed we keep the value of Lambda fixed. If we do not want to use the regularization term, we can remove this parameter.
- regularization_lambda: If we set the regularization_mode to "fixed", we need to set this parameter to the value of Lambda we want to use.
- momentum: this is needed only if we use the "tunable" regularization mode. It is the momentum to use for the update of the Lambda.
- optimizer: the optimizer to use for the training. At the moment we only support "adam" and "sgd".
- alpha: the value of the alpha parameter to use to update the Lambda. This is needed only if we use the "tunable" regularization mode.
- weight_decay_alpha: the value of the alpha parameter to use to update the weight decay. This is needed only if we use the "tunable" regularization mode.

```python
train_parameters = RegularizationConfig(
    epochs=epochs, 
    device="cuda",
    batch_size=batch_size,
    seed=seed,
    regularization = True, 
    target = 0.05,
    regularization_mode="tunable",
    momentum=0.657142498624442,
    optimizer="adam",
    alpha=0.1,
)
```

Then we have to convert the model that we want to train to a private model using Opacus:

```python
private_model, private_optimizer, private_train_loader = ModelUtils.create_private_model(
            model=model,
            epsilon=None,
            noise_multiplier=0,
            original_optimizer=optimizer,
            train_loader=train_loader,
            epochs=epochs,
            delta=0,
            MAX_GRAD_NORM=10000000000, # since we just need to wrap the model without using privacy we use a high value here
            batch_size=batch_size,
        )
```

In this case we do not want to use DP so we use a high value for the MAX_GRAD_NORM parameter and we can just put noise_multiplier=0.

Then we can train the model using the following code:

```python
for epoch in range(0, epochs):
    # Now we can train the model. First of all we will train a model without any 
    # fairness mitigation
    results = Learning.train_private_model(
            train_parameters=train_parameters,
            model=private_model,
            model_regularization=private_model_regularization,
            optimizer=private_optimizer,
            optimizer_regularization=private_optimizer_regularization,
            train_loader=private_train_loader,
            test_loader=test_loader,
            average_probabilities=None,
            current_epoch=epoch,
    )
    print(f"Epoch {epoch} - Train accuracy {results['Train Accuracy']} - Train Loss {results['Train Loss']} - Max Disparity Train {results['Max Disparity Train']}")
```


# Examples of usage

In the [example]() folder you can find two examples of usage of the code:

- CentralisedDebias: this is an example of how this library can be used to reduce the unfairness of a model trained in a classic centralised ML setting. Inside this folder you can find the notebook [CentralisedDebias.ipynb](), which contains the code to train a model on the dutch dataset while reducing the unfairness with the Demographic Parity Loss.
The notebook shows the training of a model without and with the Demographic Parity Loss, and the comparison of the fairness metrics of the two models.
- FederatedLearningDebias: this is an example of how this library can be used to reduce the unfairness of a model trained in a federated learning setting. Inside this folder you can find the notebook [fl_debias.ipynb](), which contains the code to train a model on the dutch dataset while reducing the unfairness with the Demographic Parity Loss. In the same folder, we also provide an example of how to run an hyperparameter search using Wandb. You can find a file called prob_tunable_private.ipynb that shows how to run a hyperparameter search using the library [wandb](https://wandb.ai/). To run the hyperparameter search you need to have a wandb account and to have installed the wandb library. Then running the script:
    
```
sh run_sweep.sh 
```
should be enough to start the process.

In order to start the FederatedLearningDebias examples you need to install flower

```
poetry run pip install -e git+https://github.com/lucacorbucci/flower.git@main#egg=flwr[simulation]
```

flwr = { git = "https://github.com/lucacorbucci/flower", extras = ["simulation"], branch="main" }


poetry run pip install -e git+https://github.com/lucacorbucci/flower.git@main#egg=flwr[simulation]