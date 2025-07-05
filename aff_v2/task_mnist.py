from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)


def get_transforms():
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    return apply_transforms


fds = None

"""
Load Data, apply transformation and partitioning for FashionMNIST.
"""
def load_data(partition_id: int, num_partitions: int, alpha_dirichlet: float, seed: int):
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha_dirichlet, seed=seed
        )
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    partition_train_test = partition_train_test.with_transform(get_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True, drop_last=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


"""
Train method for AFF, HETAAFF and FEDAVG.
"""
def train(net, trainloader, epochs, device, lr):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

"""
Train method for Critical-FL.
"""
def train_with_gradient_norms(net, trainloader, epochs, device, lr):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    gradient_norms = []
    losses = []
    
    for epoch in range(epochs):
        epoch_loss_avg = 0.0
        sum_norms = 0.0
        batch_count = 0
        
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            predictions = net(images)
            loss = criterion(predictions, labels)
            loss.backward()
            
            gradients = []
            for param in net.parameters():
                if param.grad is not None:
                    gradients.append(torch.flatten(param.grad))
            
            if gradients: 
                gradient_norm = torch.norm(torch.cat(gradients))
                sum_norms += gradient_norm.item()
            else:
                sum_norms += 0.0
            
            optimizer.step()
            epoch_loss_avg += loss.item()
            batch_count += 1
            
        if batch_count > 0:
            gradient_norms.append(sum_norms / batch_count)
            losses.append(epoch_loss_avg / batch_count)
    
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_gradient_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0.0
    
    local_fgn = avg_gradient_norm * lr
    
    return avg_loss, local_fgn


"""
Common test method for all strategies.
"""
def test(net, testloader, device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


"""
Get weights from model.
"""
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


"""
Set weights to model.
"""
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
