import argparse
import logging
from datetime import datetime
import torch.multiprocessing as mp
import os
from collections import defaultdict
import json
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import queue as q


from client import Client
from utils import get_dataset, calculate_divergence, get_labellist, get_model, validate_model_detailed
from partitiondata import partition_data
from partitionsystem import partition_system
from clusterclients import cluster_clients

# Setting and add additional arguments(args.testdataloader, args.labellist)
def setting(args):

    # make folder to log training, and basic logger to keep track overall training
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(timestamp, exist_ok=True)
    args.timestamp = timestamp
    logging.basicConfig(
        filename = f"{timestamp}/overall_training.log",
        filemode = "w",
        level=logging.INFO,
        format="%(message)s"
    )

    # fix randomness
    seed = args.randomseed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  

    # define list to keep track of accuracy and loss for central server and each cluster
    args.centralservertimepast = []
    args.centralserveraccuracy = []
    args.centralserverloss =[]
    args.centralserverround = []

    # set data for training and testing
    args.traindataset, testdataset = get_dataset(args.datasetname, args)

    # Set testdataloader in arguments for examining accuracy for the global distribution
    args.testdataloader = DataLoader(testdataset, batch_size=128, shuffle=False)

    # model = get_model(args.modelname)
    # validate_model_detailed(model, args.testdataloader,args)

    # Set label list
    args.labellist = get_labellist(args.datasetname)

# Create list of clients
def create_client(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist, args):
    
    # Create list of clients
    clientlist = []

    # Create clients, -1 for the clusterid  before clustering
    for i in range(args.clientnum):
        client = Client(-1, -1, clientdataloaderlist[i], clientcommunicationtimelist[i], clientcomputationtimelist[i], i, args.localepoch,args)
        clientlist.append(client)
    
    return clientlist

# Log arguments for the experiment
def logsetting(args):

    # print arguments
    # aggregation type
    intraclusteringinfo = f", intraasyncalpha : {args.intraasyncalpha}, intraasyncthreshold : {args.intraasyncthreshold}" if args.intraclusteringtype == "async" else ""
    logging.info(f"Intraclustering type : {args.intraclusteringtype}{intraclusteringinfo}")
    interclusteringinfo = f", interasyncalpha : {args.interasyncalpha}, interasyncthreshold : {args.interasyncthreshold}" if args.interclusteringtype == "async" else ""
    logging.info(f"Interclustering type : {args.interclusteringtype}{interclusteringinfo}")
    # model and dataset for training including how its partitioned to clients. for specific dataset ex.femnist, the number of clients night be fixed
    logging.info(f"Model name : {args.modelname}, Dataset name : {args.datasetname}")
    logging.info(f"Client num : {args.clientnum}")
    datainfo1 = f", Dominant percentage : {args.dominantpercentage}" if args.dataheterogeneitytype == "onelabeldominant" else ""
    datainfo2 = f", Label per client : {args.labelperclient}" if args.dataheterogeneitytype == "onlyspecificlabel" else ""
    datainfo3 = f", Dirichletalpha: {args.dirichletalpha}" if args.dataheterogeneitytype == "dirichletdistribution" else ""
    if args.datasetname not in ("femnist", "shakespeare"): logging.info(f"Data heterogeneity type : {args.dataheterogeneitytype}{datainfo1}{datainfo2}{datainfo3}")
    # system heterogeneity
    logging.info(f"System heterogeneity : {args.systemheterogeneity}")
    # how to cluster
    clusteringinfo = f", Cluster num : {args.clusternum}, Cluster size : {args.clustersize}" if args.clusteringtype == "clusterbyclientorder" or args.clusteringtype == "clusterbyrandomshuffle" else ""
    logging.info(f"Clustering type : {args.clusteringtype}{clusteringinfo}")
    logging.info(f"Cluster communication time : {args.clustercommunicationtime}")
    # how to choose epoch for each client, cluster, and centralserver
    logging.info(f"Central server epoch : {args.centralserverepoch}")
    clusterepochinfo = f", Cluster epoch : {args.clusterepoch}" if args.clusterepochtype == "fixed" else f", Cluster epoch : {args.clusterepoch}"
    logging.info(f"Cluster epoch type : {args.clusterepochtype},{clusterepochinfo}")
    localepochinfo = f", local epoch : {args.localepoch}" if args.localepochtype == "fixed" else f", local epoch : {args.localepoch}"
    logging.info(f"Local epoch type : {args.localepochtype},{localepochinfo}")
    # details
    logging.info(f"Optimizer name : {args.optimizername}, Lr : {args.learningrate}, Batch size : {args.batchsize}, Random seed : {args.randomseed}, Device : {args.device}, Regularizationcoefficient : {args.regularizationcoefficient}, Clusterparticipationratio : {args.clusterparticipationratio}, Clientparticipationratio : {args.clientparticipationratio}")

# Log client information) and save data distribution for each client in args.labelpercentageperclient
def logclient(clientlist, args):

    # logger for client
    clientlogger = logging.getLogger("client")
    handler = logging.FileHandler(f"{args.timestamp}/client_setting.log", mode = "w")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    clientlogger.addHandler(handler)
    clientlogger.propagate = False

    #logging client's data in clientlist
    for client in clientlist:
        clientdataloader = client.dataloader
        clientlogger.info(f"Unique id : {client.uniqueid}, dataset size : {len(clientdataloader.dataset)}")
        clientlogger.info(f"Communication time : {client.communicationtime}, Computation time per batch : {client.computationtimeperbatch}, Training time : {client.calculate_training_time()}")
        cachelabels = []
        for _, labels in clientdataloader:
            cachelabels.extend(labels.tolist())
        labelpercentageperclient = defaultdict(float)
        for label in args.labellist:
            count = cachelabels.count(label)
            percentage = count / len(cachelabels) if cachelabels else 0.0
            labelpercentageperclient[label] = percentage
        clientlogger.info(f"{[f'{label}:{(percentage*100):.2f}%' for label, percentage in labelpercentageperclient.items()]}")
        divergence = calculate_divergence(args.labelpercentageforglobaldistribution, labelpercentageperclient, args)
        clientlogger.info(f"Divergence : jsd {divergence['jsd']:.4f}, tvd {divergence['tvd']:.4f}")
        first_batch = next(iter(clientdataloader))
        _, labels = first_batch
        clientlogger.info(f"labels from the first batch :{[i.item() for i in labels]}")
        clientlogger.info("")

# Log cluster information
def logcluster(centralserver, args):

    # logger for cluster setting
    clusterlogger = logging.getLogger("cluster")
    handler = logging.FileHandler(f"{args.timestamp}/cluster_setting.log", mode = "w")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    clusterlogger.addHandler(handler)
    clusterlogger.propagate = False

    # log cluster information
    for cluster in centralserver.clusterlist:
        clusterlogger.info(f"Cluster {cluster.clusterid} with {len(cluster.clientlist)} clients")
        clusterlogger.info(f"Cluster communication time : {cluster.communicationtime}")
        clusterlogger.info(f"Cluster clients : {[client.uniqueid for client in cluster.clientlist]}")

        # log the cluster size data distribution and divergence
        labelpercentage = {label : 0 for label in args.labellist}
        dataincluster = 0
        for client in cluster.clientlist:
            dataincluster += len(client.dataloader.dataset)
            for label in args.labellist:
                labelpercentage[label] += len([i for _,labels in client.dataloader for i in labels if i == label])
        for label in args.labellist :
            labelpercentage[label] /= dataincluster
        divergence = calculate_divergence(args.labelpercentageforglobaldistribution, labelpercentage, args)
        formatted = [f'{label} : {labelpercentage[label] * 100:.2f}%' for label in args.labellist]
        clusterlogger.info(f"Label percentage: {formatted}%")
        clusterlogger.info(f"Divergence : jsd {divergence['jsd']:.4f}, tvd {divergence['tvd']:.4f}")
        clusterlogger.info("")

# Save graph data
def savegraph(args):

    with open(f"{args.timestamp}/args.json",'w') as f:
        graphdata = defaultdict(list)
        graphdata["centralservertimepast"] = args.centralservertimepast
        graphdata["centralserveraccuracy"] = args.centralserveraccuracy
        graphdata["centralserverloss"] = args.centralserverloss
        graphdata["centralserverround"] = args.centralserverround
        json.dump(graphdata,f)

# Main function
if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    # aggregation type for inter and intra cluster
    parser.add_argument("-intraclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    parser.add_argument("-interclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    # model and dataset for training including how its partitioned to clients. for specific dataset ex.femnist, the number of clients night be fixed
    parser.add_argument("-modelname", type = str, choices = ["cnnmnist", "cnncifar10","cnnfemnist"], default = "cnncifar10")
    parser.add_argument("-datasetname", type = str, choices = ["mnist", "cifar10", "femnist", "shakespeare"], default = "cifar10")
    parser.add_argument("-clientnum", type = int, default = 100)
    parser.add_argument("-dataheterogeneitytype", type = str, choices = ["iid", "onelabeldominant", "onlyspecificlabel", "dirichletdistribution"], default="dirichletdistribution")
    # how communication and computation in formed for clients
    parser.add_argument("-systemheterogeneity", type = str, choices = ["alltimesame", "communicationtimesamecomputationdifferent","realistic", "custom"], default = "communicationtimesamecomputationdifferent")
    # how to cluster
    parser.add_argument("-clusteringtype", type = str, choices = ["clusterbyclientorder", "clusterbyrandomshuffle", "clusterbygradientsimilarity", "custom"], default = "clusterbyclientorder")
    parser.add_argument("-clusternum", type = int, default = 100)
    parser.add_argument("-clustersize", type = int, default = 1)
    parser.add_argument("-clustercommunicationtime", type = int, default = 800)
    # how to choose epoch for each client, cluster, centralserver
    parser.add_argument("-centralserverepoch", type = int, default = 100)
    parser.add_argument("-clusterepochtype", type = str, choices = ["fixed", "custom"], default = "fixed")
    parser.add_argument("-clusterepoch", type = int, default = 1)
    parser.add_argument("-localepochtype", type =str, choices=["fixed", "custom"], default ="fixed")
    parser.add_argument("-localepoch", type = int, default = 3)
    # details
    parser.add_argument("-intraasyncalpha", type = float, default = 0.6)
    parser.add_argument("-intraasyncthreshold", type = int, default = 10)
    parser.add_argument("-interasyncalpha", type = float, default = 0.6)
    parser.add_argument("-interasyncthreshold", type = int, default = 10)
    parser.add_argument("-optimizername", type = str, default = "sgd")
    parser.add_argument("-learningrate", type = float, default = 0.01)
    parser.add_argument("-batchsize", type = int, default =32)
    parser.add_argument("-randomseed", type = int, default = 5)
    parser.add_argument("-device", type = str, default = "cuda")
    parser.add_argument("-dominantpercentage", type = int, default = 95)
    parser.add_argument("-labelperclient", type = int, default =2)
    parser.add_argument("-dirichletalpha", type = float, default = 0.1)
    parser.add_argument("-regularizationcoefficient", type =float, default = 0.0)
    parser.add_argument("-clusterparticipationratio", type = int, default = 100)
    parser.add_argument("-clientparticipationratio", type = int, default = 100)
    args = parser.parse_args()

    # make folder for to track training
    # fix randomness
    # set dataset for training and testing
    # set labellist for the dataset
    setting(args)
    logsetting(args)

    # generate clients exiting
    # generate dataset existing in each client
    # generate communicationtime and computationtime for each client
    clientdataloaderlist = partition_data(args)
    clientcommunicationtimelist, clientcomputationtimelist = partition_system(args)
    clientlist = create_client(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist,args)
    logclient(clientlist,args)


    # cluster clients in clientlist and return centralserver
    # client's clusterid, clientid, adjustment to local epoch(if needed) is set
    # cluster's clusterid, communicationtime, intraaggregationstrategy, clusterepoch is set
    # centralserver's centralserverepoch, interaggregationstrategy is set
    centralserver = cluster_clients(clientlist, args)
    logcluster(centralserver,args)

    # # # # training process
    centralserver.central_train()

    # # # save graph file
    savegraph(args)

