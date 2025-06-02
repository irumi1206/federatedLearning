import argparse
import logging
from datetime import datetime
import torch.multiprocessing as mp
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from client import Client
from cluster import Cluster
from centralserver import CentralServer
from utils import get_test_dataloader, calculate_divergence, create_loggers
from partitiondata import partition_data
from partitionsystem import partition_system
from clusterclients import clusterclients

# Setting and add additional arguments(args.testdataloader, args.labellist)
def setting(args):

    # set logging
    # make timestamp folder, args.loggers for logging each cluster, basic logger for the overall training process
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(timestamp, exist_ok=True)
    args.loggers = create_loggers(timestamp, args.clusternum)
    args.timestamp = timestamp
    logging.basicConfig(
        filename = f"{timestamp}/overall_training.log",
        filemode = "w",
        level=logging.INFO,
        format="%(message)s"
    )

    # define list to keep track of accuracy and loss for central server and each cluster
    args.centralservertimepast = []
    args.centralserveraccuracy = []
    args.centralserverloss =[]
    args.centralserverround = []
    args.clustertimepast = [[] for _ in range(args.clusternum)]
    args.clusteraccuracy = [[] for _ in range(args.clusternum)]
    args.clusterloss = [[] for _ in range(args.clusternum)]
    args.clusterround = [[] for _ in range(args.clusternum)]

    # Set multiprocessing
    mp.set_start_method("spawn", force=True)
    
    # Set testdataloader in arguments for examining accuracy for the global distribution
    args.testdataloader = get_test_dataloader(args.datasetname)

    # Set label list
    args.labellist = sorted({label.item() for _, labels in args.testdataloader for label in labels})

    # Error handling if the cluster structure does not match the number of clients
    if args.clientnum != args.clustersize*args.clusternum:
        raise ValueError("clients not matching cluster size and number")

# Create list of clients
def create_clientlist(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist, args):
    
    # Create list of clients
    clientlist = []

    # Create clients, -1 for the clusterid  before clustering
    for i in range(args.clientnum):
        client = Client(-1, -1, clientdataloaderlist[i], clientcommunicationtimelist[i], clientcomputationtimelist[i], i, args)
        clientlist.append(client)
    
    return clientlist

# Log arguments for the experiment
def logsetting(args):

    # print arguments
    logging.info("Starting cluster federated learning")
    logging.info(f"Cluster num : {args.clusternum}, Cluster size : {args.clustersize}, Client num : {args.clientnum}")
    logging.info(f"Central server epoch : {args.centralserverepoch}, Cluster server epoch : {args.clusterepoch}, Local epoch : {args.localepoch}")
    asyncinfo1 = f", Intercluseringasyncalpha: {args.interasyncalpha}" if args.interclusteringtype == "async" else ""
    asyncinfo2 = f", Intracluseringalpha: {args.intraasyncalpha}" if args.intraclusteringtype == "async" else ""
    logging.info(f"Intra clustering type : {args.intraclusteringtype}, Inter clustering type : {args.interclusteringtype}{asyncinfo1}{asyncinfo2}")
    datainfo1 = f", Dominant percentage : {args.dominantpercentage}" if args.dataheterogeneitytype == "onelabeldominant" else ""
    datainfo2 = f", Label per client : {args.labelperclient}" if args.dataheterogeneitytype == "onlyspecificlabel" else ""
    datainfo3 = f", Dirichletalpha: {args.dirichletalpha}" if args.dataheterogeneitytype == "dirichletdistribution" else ""
    logging.info(f"Data heterogeneity type : {args.dataheterogeneitytype}{datainfo1}{datainfo2}{datainfo3}")
    logging.info(f"Model name : {args.modelname}, Dataset name : {args.datasetname}, Optimizer name : {args.optimizername}, Learning rate : {args.learningrate}, Batch size : {args.batchsize}")
    logging.info(f"Random seed : {args.randomseed}")

# Log client information) and save data distribution for each client in args.labelpercentageperclient
def logclient(clientlist, args):

    # logger for client
    clientlogger = logging.getLogger("client")
    handler = logging.FileHandler(f"{args.timestamp}/client_setting.log", mode = "w")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    clientlogger.addHandler(handler)
    clientlogger.propagate = False

    # logging client's data partition, communication time, and computation time
    for i in range(args.clientnum):
        clientdataloader = clientlist[i].dataloader
        clientlogger.info(f"Client {i} with {len(clientdataloader.dataset)} data")
        clientlogger.info(f"Communication time : {clientlist[i].communicationtime}, Computation time per epoch : {clientlist[i].computationtime}, Training time : {clientlist[i].calculate_training_time()}")
        percentstring = ""
        for label in args.labellist:
            percentstring += f"{label} : {(args.labelpercentageperclient[i][label]*100):.2f}%  "
        clientlogger.info(percentstring)
        divergence = calculate_divergence(args.labelpercentageforglobaldistribution, args.labelpercentageperclient[i], args)
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
        clustertrainingtime = cluster.calculate_training_time()
        clusterlogger.info(f"Cluster computation time : {clustertrainingtime - 2*cluster.communicationtime}")
        clusterlogger.info(f"Cluster training time : {clustertrainingtime}")
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
        graphdata["clustertimepast"] = args.clustertimepast
        graphdata["clusteraccuracy"] = args.clusteraccuracy
        graphdata["clusterloss"] = args.clusterloss
        graphdata["clusterround"] = args.clusterround
        json.dump(graphdata,f)

# Main function
if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-clusternum", type = int, default = 5)
    parser.add_argument("-clustersize", type = int, default = 5)
    parser.add_argument("-clientnum", type = int, default = 25)
    parser.add_argument("-centralserverepoch", type = int, default = 100)
    parser.add_argument("-clusterepoch", type = int, default = 2)
    parser.add_argument("-localepoch", type = int, default = 5)
    parser.add_argument("-intraclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    parser.add_argument("-interclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    parser.add_argument("-modelname", type = str, choices = ["cnnmnist", "cnncifar10"], default = "cnnmnist")
    parser.add_argument("-datasetname", type = str, choices = ["mnist", "cifar10"], default = "mnist")
    parser.add_argument("-dataheterogeneitytype", type = str, choices = ["iid", "onelabeldominant", "onlyspecificlabel", "dirichletdistribution"], default="iid")
    parser.add_argument("-clusteringtype", type = str, choices = ["clusterbyclientorder", "clusterbyrandomshuffle", "clusterbysimilarlabel", "clusterbysimilarsystem"], default = "clusterbyclientorder")
    parser.add_argument("-systemheterogeneity", type = str)
    parser.add_argument("-intraasyncalpha", type = float, default = 0.6)
    parser.add_argument("-interasyncalpha", type = float, default = 0.6)
    parser.add_argument("-optimizername", type = str, default = "sgd")
    parser.add_argument("-learningrate", type = float, default = 0.01)
    parser.add_argument("-batchsize", type = int, default =32)
    parser.add_argument("-randomseed", type = int, default = 1)
    parser.add_argument("-device", type = str, default = "cuda")
    parser.add_argument("-dominantpercentage", type = int, default = 95)
    parser.add_argument("-labelperclient", type = int, default =2)
    parser.add_argument("-dirichletalpha", type = float, default = 0.1)
    args = parser.parse_args()

    # set logger for the general training process
    # set logger for eahc cluster
    # setting for the multi proecessing in sync setting
    # set args.testdataloader for the validation across global distribution
    # set args.labellist for the label list
    setting(args)
    
    # set communication and system time for each cluster and client
    # split the data and distribute to each clients as a list. and also save the data distribution for each client in args.labelpercentageperclient, and global data distribution in args.labelpercentageforglobaldistribution
    # create list of clients
    clustercommunicationtimelist, clientcommunicationtimelist, clientcomputationtimelist = partition_system(args)
    clientdataloaderlist = partition_data(args)
    clientlist = create_clientlist(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist, args)

    # clustering

    centralserver = clusterclients(clientlist, clustercommunicationtimelist, args)

    # log experiment setting(args)
    # log client information
    # log the cluster information
    logsetting(args)
    logclient(clientlist,args)
    logcluster(centralserver,args)

    # training process
    centralserver.central_train()

    # save graph file
    savegraph(args)

