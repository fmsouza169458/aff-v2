import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigRecord
import os

DATASET = os.getenv("DATASET", "MNIST")
if DATASET == "CIFAR10":
    from aff_v2.task_cifar import Net, get_weights, load_data, set_weights, test, train, train_with_gradient_norms
else:
    from aff_v2.task_mnist import Net, get_weights, load_data, set_weights, test, train, train_with_gradient_norms

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
        self.strategy = os.getenv("STRATEGY", "AFF_V2")

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    """
    Using the strategy and dataset provided on a training section,
    selects the appropriate train method
    """
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        if self.strategy == "CRITICAL_FL":
            if self.dataset == "CIFAR10":
                train_metrics = train_with_gradient_norms(
                    self.net,
                    self.trainloader,
                    self.valloader,
                    epochs=1,
                    learning_rate=self.learning_rate,
                    device=self.device,
                )
                train_loss = train_metrics["train_loss"]
                local_fgn = train_metrics.get("local_fgn", 0.0)
            else:
                train_loss, local_fgn = train_with_gradient_norms(
                    self.net,
                    self.trainloader,
                    self.local_epochs,
                    device=self.device,
                    lr=self.learning_rate,
                )
            
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {
                    "train_loss": train_loss,
                    "local_fgn": local_fgn,
                },
            )
        else:
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
                    lr=self.learning_rate,
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
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


"""
Initializes the client using the parameters provided for the experiment
"""
def client_fn(context: Context):
    seed = int(os.getenv("SEED", "1"))
    set_seed(seed)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    alpha_dirichlet = float(os.getenv("ALPHA", "0.3"))
    dataset = os.getenv("DATASET", "MNIST")
    
    if dataset == "CIFAR10":
        batch_size = context.run_config["batch-size"]
        trainloader, valloader = load_data(partition_id, num_partitions, batch_size, alpha_dirichlet, seed)
    else:
        trainloader, valloader = load_data(partition_id, num_partitions, alpha_dirichlet, seed)
    
    local_epochs = context.run_config["local-epochs"]

    return FlowerClient(trainloader, valloader, local_epochs, context).to_client()


app = ClientApp(client_fn=client_fn)