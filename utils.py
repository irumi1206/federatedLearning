import importlib
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import logging
import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import defaultdict
from datasets import load_dataset
from random import seed,shuffle
from dataset.FEMNISTClientDataset import FEMNISTClientDataset
from torch.utils.data import Subset
import random

# Return model object based on the model name
def get_model(modelname):

    if modelname == "cnnmnist":
        module = importlib.import_module(f"models.{modelname}")
        return module.get_model()
    elif modelname == "cnncifar10":
        module = importlib.import_module(f"models.{modelname}")
        return module.get_model()
    elif modelname == "cnnfemnist":
        module = importlib.import_module(f"models.{modelname}")
        return module.get_model()
    else :
        raise ValueError("model not supported")

# Return dataset for training and testing
def get_dataset(datasetname,args):

    if datasetname == "mnist":
        traindataset = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())      
        testdataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    elif datasetname == "cifar10":
        traindataset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transforms.ToTensor() )
        testdataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    elif datasetname == "femnist":

        # get the dataset
        dataset = load_dataset("flwrlabs/femnist", split = "train")

        # get clietnum + testing clients from the dataset
        writer = dataset.unique("writer_id")
        testratio = 0.2
        clientnumtosample = args.clientnum + int(args.clientnum*testratio)
        random.seed(args.randomseed)
        sampledwriter = random.sample(writer,clientnumtosample)
        testwriter = sampledwriter[args.clientnum:]
        trainwriter = sampledwriter[:args.clientnum]
        testwriterset = set(testwriter)
        trainwriterset = set(trainwriter)

        testdataset = dataset.filter(lambda example: example["writer_id"] in testwriterset)
        traindataset = dataset.filter(lambda example: example["writer_id"] in trainwriterset)

        imagetransform = transforms.Compose([
            transforms.ToTensor()
        ])

        testdataset.set_transform(
            lambda batch: {
                'image': [imagetransform(img) for img in batch['image']],
                'writer_id': batch['writer_id'],
                'character': batch['character']
                # Add any other columns you want to preserve here
            }
        )
        traindataset.set_transform(
            lambda batch: {
                'image': [imagetransform(img) for img in batch['image']],
                'writer_id': batch['writer_id'],
                'character': batch['character']
                # Add any other columns you want to preserve here
            }
        )

        print(len(testdataset))
        
        print(len(traindataset))

        testdataset = FEMNISTClientDataset(testdataset)
    
    else:
        raise ValueError("datasetname not supported")
    
    return traindataset,testdataset

def get_labellist(datasetname):

    if datasetname == "mnist":
        return [i for i in range(10)]
    elif datasetname == "cifar10":
        return [i for i in range(10)]
    elif datasetname == "femnist":
        return [i for i in range(62)]
    elif datasetname == "shakespeare":
        return [i for i in range(81)]
    else:
        raise ValueError("datasetname not supported")

# Return optimizer object based on the modle object, optimizer name, and learning rate
def get_optimizer(model, optimizername, learningrate):

    if optimizername == "sgd":
        return optim.SGD(model.parameters(), lr =learningrate)

    else:
        raise ValueError("optimizer not supported yet")

# Return loss and accuracy based on the model object, data loader, and device
def validate_model(model, dataloader, args):

    model.eval()
    correct = 0
    total = 0
    lossfunction = nn.CrossEntropyLoss()
    totalloss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = lossfunction(outputs, labels)
            totalloss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = totalloss / total
    return accuracy, avg_loss

# Return more detailed infomatin about the accuracy for each label
def validate_model_detailed(model, dataloader, args):

    model.eval()
    correct = 0
    total = 0
    lossfunction = nn.CrossEntropyLoss()
    totalloss = 0.0
    correctperlabel = defaultdict(int)
    incorrectperlabel = defaultdict(int)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = lossfunction(outputs, labels)
            totalloss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            correctlabels = [label for label, pred in zip(labels, predicted) if label == pred]
            incorrectlabels = [label for label, pred in zip(labels, predicted) if label != pred]
            for label in correctlabels:
                correctperlabel[label.item()] += 1
            for label in incorrectlabels:
                incorrectperlabel[label.item()] += 1

    accuracyperlabel = defaultdict(float)
    for label in args.labellist:
        if correctperlabel[label] + incorrectperlabel[label] != 0:
            accuracyperlabel[label] = correctperlabel[label] / (correctperlabel[label] + incorrectperlabel[label])
        else:
            accuracyperlabel[label] = 0.0

    accuracy = correct / total
    avg_loss = totalloss / total
    return accuracy, avg_loss, accuracyperlabel

# Calculate jsd and tvd given global and local distribution in dictionary format
def calculate_divergence(globaldistribution, localdistribution, args):

    # make array from dictionary
    globalarray = np.array([globaldistribution[label] for label in args.labellist])
    localarray = np.array([localdistribution[label] for label in args.labellist])

    # normalize
    globalarray /= globalarray.sum()
    localarray /= localarray.sum()

    # calculate jsd and tvd
    jsd = jensenshannon(globalarray, localarray, base = 2) ** 2
    tvd = 0.5 * np.sum(np.abs(globalarray - localarray))

    return {"jsd":jsd, "tvd":tvd}

# Create logger for each cluster
def create_loggers(dirname, clusternum):
    
    # Create directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)
    loggers = []
    
    for i in range(clusternum):
        
        name = f"cluster_{i}"
        filepath = os.path.join(dirname, f"{name}.log")
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.FileHandler(filepath, mode = "w")
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
        
        loggers.append(logger)
        
    return loggers