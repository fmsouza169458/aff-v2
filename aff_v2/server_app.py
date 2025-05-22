import json
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader
from datasets import load_dataset

#from aff_v2.task_cifar import Net, get_weights, set_weights, test, get_transforms
from aff_v2.task_mnist import Net, get_weights, set_weights, test, get_transforms
#from aff_v2.task import Net, get_weights, set_weights, test, get_transforms

from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10

from aff_v2.aff import AFFStrategy
from aff_v2.aff_without_het import AFFStrategyWithoutHet
from aff_v2.fedavgg_constant import FedAvgWithLogging

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model."""
    def evaluate(server_round, parameters_ndarrays, config):
        # Instantiate model
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        # Run test
        loss, accuracy = test(net, testloader, device)
        return loss, {"cen_accuracy": accuracy}
    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return {"max_b": max(b_values)}


def on_fit_config(server_round: int) -> Metrics:
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Inicializa os pesos do modelo
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Carrega conjunto de teste
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)
   
    
    strategy = AFFStrategy(
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        min_available_clients=100,
        max_clients=100,
        initial_clients=10,
    )

    """ # Prepare initial model parameters
    model = Net()
    model_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(model_params)

    # Define strategy
    strategy = FedAvgWithLogging(
        fraction_fit=0.1,
        min_available_clients=100,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
        on_fit_config_fn=on_fit_config,
        initial_parameters=parameters,
    ) """

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Cria o servidor com AFF
app = ServerApp(server_fn=server_fn)