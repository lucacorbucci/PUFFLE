import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import wandb
from fl_puf.Utils.utils import Utils
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from Regularization.DPL import RegularizationLoss


class Learning:
    @staticmethod
    def create_private_model(
        model,
        epsilon,
        optimizer,
        train_loader,
        epochs,
        delta,
        MAX_GRAD_NORM,
    ):
        print(epsilon)
        print(delta)
        print(MAX_GRAD_NORM)
        privacy_engine = PrivacyEngine(accountant="rdp")
        if epsilon:
            (
                private_model,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=MAX_GRAD_NORM,
            )
        else:
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=0,
                max_grad_norm=MAX_GRAD_NORM,
            )

        return private_model, optimizer, train_loader

    @staticmethod
    def train_loop(
        epochs: int,
        model,
        model_regularization,
        optimizer,
        optimizer_regularization,
        train_loader,
        test_loader,
        device,
        private,
        criterion,
        DPL,
        DPL_lambda,
        loss_lambda=1.0,
    ):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(15)

        for epoch in range(1, epochs + 1):
            Learning.train_model(
                model,
                model_regularization,
                device,
                train_loader,
                optimizer,
                optimizer_regularization,
                epoch,
                DPL,
                DPL_lambda,
                loss_lambda,
                private,
            )

            (
                test_loss,
                accuracy,
                max_disparity_test,
            ) = Learning.test(model, test_loader, device, epoch, DPL_lambda)

    @staticmethod
    def compute_regularization_term(
        data,
        target,
        sensitive_feature,
        model_regularization,
        device,
        criterion_regularization,
        DPL_lambda,
        private,
    ):
        """This function computes the regularization term on the training data
        passed as parameter

        Args:
            data (_type_): the dataset on which the regularization term is computed
            target (_type_): the targets of the data we pass as parameter
            sensitive_feature (_type_): the corresponding sensitive features
            model (_type_): the model we are using to compute the regularization term
                This is the same model that we are optimizing using the result of the
                regularization term
            device (_type_): GPU/CPU
        """
        outputs = model_regularization(data)

        possible_targets = set([item.item() for item in target])
        sensitive_attributes = set([item.item() for item in sensitive_feature])

        fairness_violation = criterion_regularization(
            sensitive_attribute_list=sensitive_feature,
            targets=target,
            device=device,
            predictions=outputs,
            sensitive_attributes=list(sensitive_attributes),
            possible_targets=list(possible_targets),
        )

        regularization_term = fairness_violation  # DPL_lambda * fairness_violation
        if private:
            regularization_term.backward()
        return regularization_term, fairness_violation

    @staticmethod
    def train(
        model_regularization,
        model,
        device,
        sensitive_feature,
        target,
        data,
        criterion_regularization,
        DPL_lambda,
        loss_lambda,
        criterion,
        optimizer,
        optimizer_regularization,
        private,
        DPL,
    ):
        running_loss = 0.0
        total_correct = 0
        running_regularization_term = 0.0
        total = 0
        # First of all we synchronize the two models so that
        # they have the same weights
        if model_regularization:
            Utils.sync_models(model_regularization, model)
        target = target.to(device)
        data = data.to(device)
        sensitive_feature = sensitive_feature.to(device)

        if DPL and private and model_regularization:
            # We use the "regularization" model to compute the output
            # and then we use the output to compute the regularization term
            # In the end we compute the backward using the regularization term
            # and we get the per sample gradient
            (
                regularization_term,
                fairness_violation,
            ) = Learning.compute_regularization_term(
                data,
                target,
                sensitive_feature,
                model_regularization,
                device,
                criterion_regularization,
                DPL_lambda,
                private,
            )
        elif DPL and not private:
            # In this case we don't need a separate model for the
            # regularization, we have to use the original model
            (
                regularization_term,
                fairness_violation,
            ) = Learning.compute_regularization_term(
                data,
                target,
                sensitive_feature,
                model,
                device,
                criterion_regularization,
                DPL_lambda,
                private,
            )
            print(regularization_term, fairness_violation)

        # Now we can compute the forward and backward pass
        # using the original model
        outputs = model(data)

        if DPL and not private:
            # If we are running a model that is not private
            # and we want to use the regularization term
            # we can just sum the two losses and then perform
            # backward on the sum of the two losses
            # DPL_lambda = DPL_lambda.to(device)

            loss = criterion(outputs, target) + DPL_lambda * fairness_violation
        else:
            # Instead, if we are running a private model
            # or if we are running a plain model without privacy
            # and regularization, then we can just compute the
            # loss using the original model. In the case of
            # private model, we have already computed the
            # backward pass on the regularization term and
            # then we will sum the per sample gradients. In the case
            # of a plain model, we will just compute the backward
            # using a classic loss function
            loss = criterion(outputs, target)

        loss.backward()

        if model_regularization and private:
            # If we want to use the regularization we have
            # to sum the gradients of the original model
            # and of the regularized model
            for p1, p2 in zip(model.parameters(), model_regularization.parameters()):
                p1.grad_sample += p2.grad_sample

        optimizer.step()
        optimizer.zero_grad()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == target).float().sum()
        running_loss += loss.item()

        if model_regularization:
            optimizer_regularization.zero_grad()
            running_loss += regularization_term.item()
            running_regularization_term += fairness_violation.item()
        total_correct += correct
        total += target.size(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return running_loss, total_correct, running_regularization_term, total

    @staticmethod
    def train_model(
        model,
        model_regularization,
        device,
        train_loader,
        optimizer,
        optimizer_regularization,
        epoch,
        DPL,
        DPL_lambda,
        loss_lambda,
        private,
    ):
        """Train the network and computes loss and accuracy.

        Raises
        ------
            Exception: Raises an exception when Federated Learning is not initialized

        Returns
        -------
            Tuple[float, float]: Loss and accuracy on the training set.
        """
        criterion = nn.CrossEntropyLoss()

        model.train()
        criterion_regularization = RegularizationLoss()

        if model_regularization:
            # If we want to use the DPL regularization then we
            # have to put the model in training mode
            model_regularization.train()
        if private:
            MAX_PHYSICAL_BATCH_SIZE = 512
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer,
            ) as memory_safe_data_loader:
                for batch_index, (data, sensitive_feature, target) in enumerate(
                    memory_safe_data_loader, 0
                ):
                    (
                        running_loss,
                        total_correct,
                        running_regularization_term,
                        total,
                    ) = Learning.train(
                        model_regularization,
                        model,
                        device,
                        sensitive_feature,
                        target,
                        data,
                        criterion_regularization,
                        DPL_lambda,
                        loss_lambda,
                        criterion,
                        optimizer,
                        optimizer_regularization,
                        private,
                        DPL,
                    )
        else:
            for batch_index, (data, sensitive_feature, target) in enumerate(
                train_loader, 0
            ):
                (
                    running_loss,
                    total_correct,
                    running_regularization_term,
                    total,
                ) = Learning.train(
                    model_regularization,
                    model,
                    device,
                    sensitive_feature,
                    target,
                    data,
                    criterion_regularization,
                    DPL_lambda,
                    loss_lambda,
                    criterion,
                    optimizer,
                    optimizer_regularization,
                    private,
                    DPL,
                )

        # Learning.evaluate_model(
        #     running_loss,
        #     train_loader,
        #     total_correct,
        #     total,
        #     model_regularization,
        #     running_regularization_term,
        #     epoch,
        #     device,
        #     criterion_regularization,
        #     model,
        #     wandb_run,
        #     DPL_lambda,
        # )

    @staticmethod
    def test(
        model,
        test_loader,
        device,
        epoch,
        set_name="test set",
        DPL_lambda=None,
        max_disparity_computation=True,
    ):
        """_summary_

        Args:
            model (_type_): the model to test
            test_loader (_type_): the test dataset we want to use
            device (_type_): the device we want to use (GPU/CPU)
            wandb_run (_type_): the wandb run we want to use to log the results
            epoch (_type_): current epoch
            set_name (str, optional): the . Defaults to "test set".


        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        losses = []
        predictions = []
        colors = []

        with torch.no_grad():
            for data, color, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total += target.size(0)
                test_loss = criterion(output, target).item()
                losses.append(test_loss)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                y_pred.append(pred)
                y_true.append(target)
                predictions.append(pred)
                colors += [item.item() for item in color]

        predictions = [value.item() for item in predictions for value in item]
        counter_predictions = defaultdict(list)
        for prediction, color in zip(predictions, colors):
            counter_predictions[color].append(prediction)

        criterion_regularization = RegularizationLoss()

        max_disparity_test = None
        if max_disparity_computation:
            max_disparity_test = criterion_regularization.violation_with_dataset(
                model,
                test_loader,
                device,
            )

        test_loss = np.mean(losses)
        accuracy = correct / total

        y_true = [item.item() for sublist in y_true for item in sublist]
        y_pred = [item.item() for sublist in y_pred for item in sublist]

        return (
            test_loss,
            accuracy,
            max_disparity_test,
        )

    @staticmethod
    def get_evaluate_fn(
        testset, dataset_name: str, wandb_run: wandb.sdk.wandb_run.Run, config
    ) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""
        config = config()

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, float]]:
            """Use the entire CIFAR-10 test set for evaluation."""
            # print(">>>>>> SERVER ", config)

            # determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model = Utils.get_model(dataset_name)
            Utils.set_params(model, parameters)
            model.to(device)

            testloader = torch.utils.data.DataLoader(
                testset, batch_size=512  # config["batch_size"],
            )
            (
                loss,
                accuracy,
                max_disparity_test,
            ) = Learning.test(
                model,
                testloader,
                device=device,
                epoch=0,
                DPL_lambda=None,
            )
            print(f"Evaluation {max_disparity_test}")
            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": server_round,
                        "acc": accuracy,
                        "loss": loss,
                        "max_disparity_test": max_disparity_test,
                    },
                )
            return loss, {"accuracy": accuracy}

        return evaluate


class ModelUtils:
    # borrowed from Pytorch quickstart example
    @staticmethod
    def train(net, trainloader, epochs, device: str):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        net.train()
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()

    # borrowed from Pytorch quickstart example
    @staticmethod
    def test(net, testloader, device: str):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy
