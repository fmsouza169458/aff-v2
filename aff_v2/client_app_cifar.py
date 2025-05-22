"""custom_mods: A Flower app with custom mods."""

import torch
from aff_v2.task_cifar import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord

from random import random

import json


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, context: Context):
        self.client_state = context.state
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train_metrics = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs=1,
            learning_rate=self.lr,
            device=self.device,
        )

        train_loss = train_metrics["train_loss"]

        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            # If first entry, create the list
            if train_loss is not None:
                fit_metrics["train_loss_hist"] = [train_loss]
        else:
            # If it's not the first entry, append to the existing list
            if train_loss is not None:
                fit_metrics["train_loss_hist"].append(train_loss)

        # A complex metric strcuture can be returned by a ClientApp if it is first
        # converted to a supported type by `flwr.common.Scalar`. Here we serialize it with
        # JSON and therefore representing it as a string (one of the supported types)
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)

        return get_weights(self.net), len(self.trainloader.dataset), {
                "train_loss": train_loss,
                "my_metric": complex_metric_str,
            }, 

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate, context).to_client()


app = ClientApp(
    client_fn=client_fn,
)
