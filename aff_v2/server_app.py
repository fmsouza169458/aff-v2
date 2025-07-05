import json
import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch

from aff_v2.task_cifar import Net as cifar_net, get_weights as cifar_get_weights, set_weights as cifar_set_weights, test as cifar_test, get_transforms as cifar_get_transforms
from aff_v2.task_mnist import Net as mnist_net, get_weights as mnist_get_weights, set_weights as mnist_set_weights, test as mnist_test, get_transforms as mnist_get_transforms

from aff_v2.fedavgg_constant import FedAvgWithLogging
from aff_v2.aff_with_het import AffWithHet
from aff_v2.aff_without_het import AffWithoutHet
from aff_v2.critical_fl import CriticalFL
from aff_v2.utils import set_seed

def get_evaluate_fn(testloader, device, net_function, set_weights_function, test_function):
    def evaluate(server_round, parameters_ndarrays, config):
        net = net_function()
        set_weights_function(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test_function(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}
    return evaluate

"""
Get the experiment configuration from the environment variables
"""
def get_strategy_config() -> dict:
    dataset = os.getenv("DATASET", "MNIST")
    initial_ff = float(os.getenv("INITIAL_FF", "0.1"))
    rounds = int(os.getenv("ROUNDS", "250"))
    alpha = float(os.getenv("ALPHA", "0.3"))
    strategy_type = os.getenv("STRATEGY", "AFF_V2")
    polynomial_degree = int(os.getenv("POLYNOMIAL_DEGREE", "1"))
    max_window_size = int(os.getenv("MAX_WINDOW_SIZE", "20"))
    min_window_size = int(os.getenv("MIN_WINDOW_SIZE", "2"))
    use_heterogeneity = os.getenv("USE_HETEROGENEITY", "false").lower() == "true"
    seed = int(os.getenv("SEED", "unknown"))
    fgn_threshold = float(os.getenv("FGN_THRESHOLD", "0.01"))

    
    config = {
        "dataset": dataset,
        "rounds": rounds,
        "initial_ff": initial_ff,
        "alpha": alpha,
        "strategy_type": strategy_type,
        "polynomial_degree": polynomial_degree,
        "max_window_size": max_window_size,
        "min_window_size": min_window_size,
        "use_heterogeneity": use_heterogeneity,
        "fgn_threshold": fgn_threshold,
        "seed": seed
    }
    
    return config


def server_fn(context: Context):
    strategy_config = get_strategy_config()

    set_seed(strategy_config["seed"])

    num_rounds = strategy_config["rounds"]

    """
    This section is used to load proper dataset and functions
    And assign it to defined strategy
    """
    if strategy_config["dataset"] == "MNIST":
        ndarrays = mnist_get_weights(mnist_net())
        parameters = ndarrays_to_parameters(ndarrays)

        testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
        testloader = DataLoader(testset.with_transform(mnist_get_transforms()), batch_size=32)

        net_function = mnist_net
        set_weights_function = mnist_set_weights
        test_function = mnist_test
    elif strategy_config["dataset"] == "CIFAR10":
        ndarrays = cifar_get_weights(cifar_net())
        parameters = ndarrays_to_parameters(ndarrays)

        testset = load_dataset("uoft-cs/cifar10")["test"]
        testloader = DataLoader(testset.with_transform(cifar_get_transforms()), batch_size=32)

        net_function = cifar_net
        set_weights_function = cifar_set_weights
        test_function = cifar_test

    if strategy_config["strategy_type"] == "AFF_V2":
        if strategy_config["use_heterogeneity"]:
            strategy = AffWithHet(
                initial_parameters=parameters,
                evaluate_fn=get_evaluate_fn(testloader, device="cpu", net_function=net_function, set_weights_function=set_weights_function, test_function=test_function),
                min_available_clients=100,
                max_clients=100,
                initial_clients=int(100*strategy_config["initial_ff"]),
                degree=strategy_config["polynomial_degree"],
                max_window_size=strategy_config["max_window_size"],
                min_window_size=strategy_config["min_window_size"]
            )
        else:
            strategy = AffWithoutHet(
                initial_parameters=parameters,
                evaluate_fn=get_evaluate_fn(testloader, device="cpu", net_function=net_function, set_weights_function=set_weights_function, test_function=test_function),
                min_available_clients=100,
                max_clients=100,
                initial_clients=int(100*strategy_config["initial_ff"]),
                degree=strategy_config["polynomial_degree"],
                max_window_size=strategy_config["max_window_size"],
                min_window_size=strategy_config["min_window_size"]
            )
    elif strategy_config["strategy_type"] == "CRITICAL_FL":
        strategy = CriticalFL(
            initial_parameters=parameters,
            evaluate_fn=get_evaluate_fn(testloader, device="cpu", net_function=net_function, set_weights_function=set_weights_function, test_function=test_function),
            min_available_clients=100,
            max_clients=100,
            initial_clients=int(100*strategy_config["initial_ff"]),
            min_clients=2,
            fgn_threshold=strategy_config["fgn_threshold"]
        )
    elif strategy_config["strategy_type"] == "CONSTANT":
        strategy = FedAvgWithLogging(
            fraction_fit=strategy_config["initial_ff"],
            min_available_clients=100,
            evaluate_fn=get_evaluate_fn(testloader, device="cpu", net_function=net_function, set_weights_function=set_weights_function, test_function=test_function),
            initial_parameters=parameters,
        ) 

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Limit the number of threads used for intra-op parallelism
torch.set_num_threads(4) #4 threads in dl28 machine achieves the same performance with resource limitation
# Limit the number of threads used for inter-op parallelism (e.g., for parallel calls to different operators)
torch.set_num_interop_threads(2) #2 threads in dl28 machine achieves the same performance with resource limitation

app = ServerApp(server_fn=server_fn)