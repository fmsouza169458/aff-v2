import torch
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
import json
import os

# Importa condicionalmente baseado na variável de ambiente DATASET
DATASET = os.getenv("DATASET", "MNIST")
if DATASET == "CIFAR10":
    from aff_v2.task_cifar import Net, get_weights, load_data, set_weights, test, train
else:  # MNIST
    from aff_v2.task_mnist import Net, get_weights, load_data, set_weights, test, train

from aff_v2.utils import set_seed

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.dataset = os.getenv("DATASET", "MNIST")
        
        self.learning_rate = context.run_config.get("learning-rate", 0.01)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        
        if self.dataset == "CIFAR10":
            train_metrics = train(
                self.net,
                self.trainloader,
                self.valloader,
                epochs=1,
                learning_rate=self.learning_rate,
                device=self.device,
            )
            train_loss = train_metrics["train_loss"]
        else:
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                device=self.device,
                lr=config["lr"],
            )

        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            if train_loss is not None:
                fit_metrics["train_loss_hist"] = [train_loss]
        else:
            if train_loss is not None:
                fit_metrics["train_loss_hist"].append(train_loss)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    seed = int(os.getenv("SEED", "1"))

    set_seed(seed)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    alpha_dirichlet = float(os.getenv("ALPHA", "0.3"))
    dataset = os.getenv("DATASET", "MNIST")
    
    if dataset == "CIFAR10":
        batch_size = context.run_config["batch-size"]
        trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha_dirichlet)
    else:
        trainloader, valloader = load_data(partition_id, num_partitions, alpha_dirichlet)
    
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(trainloader, valloader, local_epochs, context).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)