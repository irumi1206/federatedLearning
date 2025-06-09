import argparse
import logging
from datetime import datetime
import torch.multiprocessing as mp
import os
from collections import defaultdict
import json
from torch.utils.data import DataLoader

from client import Client
from utils import get_test_dataset, get_train_dataset, calculate_divergence, create_loggers
from partitiondata import partition_data
from partitionsystem import partition_system
from clusterclients import clusterclients

# Setting and add additional arguments(args.testdataloader, args.labellist)
def setting(args):

    # set logging
    # make timestamp folder, args.loggers for logging each cluster, basic logger for the overall training process
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    os.makedirs(timestamp, exist_ok=True)
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

    # set multiprocessing
    mp.set_start_method("spawn", force=True)

    # set dataset for training
    args.traindataset = get_train_dataset(args.datasetname)
    
    # Set testdataloader in arguments for examining accuracy for the global distribution
    args.testdataloader = DataLoader(get_test_dataset(args.datasetname), batch_size=128, shuffle=False) 

    # Set label list
    args.labellist = sorted({label.item() for _, labels in args.testdataloader for label in labels})

# Create list of clients
def create_clientlist(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist, args):
    
    # Create list of clients
    clientlist = []

    # Create clients, -1 for the clusterid  before clustering
    for i in range(args.clientnum):
        client = Client(-1, -1, clientdataloaderlist[i], clientcommunicationtimelist[i], clientcomputationtimelist[i], i, -1,args)
        clientlist.append(client)
    
    return clientlist

# Log arguments for the experiment
def logsetting(args):

    # print arguments
    # general federated learning setting
    logging.info("Starting cluster federated learning")
    logging.info(f"Model name : {args.modelname}, Dataset name : {args.datasetname}")
    intraclusteringinfo = f", intraasyncalpha : {args.intraasyncalpha}, intraasyncthreshold : {args.intraasyncthreshold}" if args.intraclusteringtype == "async" else ""
    logging.info(f"Intraclustering type : {args.intraclusteringtype}{intraclusteringinfo}")
    interclusteringinfo = f", interasyncalpha : {args.interasyncalpha}, interasyncthreshold : {args.interasyncthreshold}" if args.interclusteringtype == "async" else ""
    logging.info(f"Interclustering type : {args.interclusteringtype}{interclusteringinfo}")
    # how to partition client's system and system
    logging.info(f"Client num : {args.clientnum}, System heterogeneity : {args.systemheterogeneity}")
    datainfo1 = f", Dominant percentage : {args.dominantpercentage}" if args.dataheterogeneitytype == "onelabeldominant" else ""
    datainfo2 = f", Label per client : {args.labelperclient}" if args.dataheterogeneitytype == "onlyspecificlabel" else ""
    datainfo3 = f", Dirichletalpha: {args.dirichletalpha}" if args.dataheterogeneitytype == "dirichletdistribution" else ""
    logging.info(f"Data heterogeneity type : {args.dataheterogeneitytype}{datainfo1}{datainfo2}{datainfo3}")
    # how to cluster and set cluster communication time
    clusteringinfo = f", Cluster num : {args.clusternum}, Cluster size : {args.clustersize}" if args.clusteringtype == "clusterbyclientorder" or args.clusteringtype == "clusterbyrandomshuffle" else ""
    logging.info(f"Clustering type : {args.clusteringtype}{clusteringinfo}")
    logging.info(f"Cluster communication time : {args.clustercommunicationtime}")
    # how to choose epoch for each client, cluster, and centralserver
    logging.info(f"Central server epoch : {args.centralserverepoch}")
    clusterepochinfo = f", Cluster epoch : {args.clusterepoch}" if args.clusterepochtype == "fixed" else ""
    logging.info(f"Cluster epoch type : {args.clusterepochtype},{clusterepochinfo}")
    localepochinfo = f", local epoch : {args.localepoch}" if args.localepochtype == "fixed" else ""
    logging.info(f"Local epoch type : {args.localepochtype},{localepochinfo}")
    # details
    logging.info(f"Optimizer name : {args.optimizername}, Lr : {args.learningrate}, Batch size : {args.batchsize}, Random seed : {args.randomseed}, Device : {args.device}, Regularizationcoefficient : {args.regularizationcoefficient}")

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
        percentstring = ""
        for label in args.labellist:
            percentstring += f"{label} : {(args.labelpercentageperclient[client.uniqueid][label]*100):.2f}%  "
        clientlogger.info(percentstring)
        divergence = calculate_divergence(args.labelpercentageforglobaldistribution, args.labelpercentageperclient[client.uniqueid], args)
        clientlogger.info(f"Divergence : jsd {divergence['jsd']:.4f}, tvd {divergence['tvd']:.4f}")
        first_batch = next(iter(clientdataloader))
        _, labels = first_batch
        clientlogger.info(f"labels from the first batch :{[i.item() for i in labels]}")
        clientlogger.info("")

    # logging client's data partition, communication time, and computation time
    # for i in range(args.clientnum):
    #     clientdataloader = clientlist[i].dataloader
    #     clientlogger.info(f"Client {i} with {len(clientdataloader.dataset)} data")
    #     clientlogger.info(f"Communication time : {clientlist[i].communicationtime}, Computation time per batch : {clientlist[i].computationtimeperbatch}, Training time : {clientlist[i].calculate_training_time()}")
    #     percentstring = ""
    #     for label in args.labellist:
    #         percentstring += f"{label} : {(args.labelpercentageperclient[i][label]*100):.2f}%  "
    #     clientlogger.info(percentstring)
    #     divergence = calculate_divergence(args.labelpercentageforglobaldistribution, args.labelpercentageperclient[i], args)
    #     clientlogger.info(f"Divergence : jsd {divergence['jsd']:.4f}, tvd {divergence['tvd']:.4f}")
    #     first_batch = next(iter(clientdataloader))
    #     _, labels = first_batch
    #     clientlogger.info(f"labels from the first batch :{[i.item() for i in labels]}")
    #     clientlogger.info("")

# Log cluster information
def logcluster(centralserver, args):

    # set loggers for clusters
    args.loggers = create_loggers(args.timestamp, len(centralserver.clusterlist))

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
        json.dump(graphdata,f)

# Main function
if __name__ == "__main__":

    # define parser
    parser = argparse.ArgumentParser()
    # general information about the cluster federated leanring
    parser.add_argument("-modelname", type = str, choices = ["cnnmnist", "cnncifar10"], default = "cnnmnist")
    parser.add_argument("-datasetname", type = str, choices = ["mnist", "cifar10"], default = "mnist")
    parser.add_argument("-intraclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    parser.add_argument("-interclusteringtype", type = str, choices = ["sync", "async"], default = "sync")
    # how to partition the device in data and system
    parser.add_argument("-clientnum", type = int, default = 10)
    parser.add_argument("-dataheterogeneitytype", type = str, choices = ["iid", "onelabeldominant", "onlyspecificlabel", "dirichletdistribution"], default="onelabeldominant")
    parser.add_argument("-systemheterogeneity", type = str, choices = ["alltimesame", "communicationtimesamecomputationdifferent","realistic", "custom"], default = "custom")
    # how to cluster
    parser.add_argument("-clusteringtype", type = str, choices = ["clusterbyclientorder", "clusterbyrandomshuffle", "custom"], default = "custom")
    parser.add_argument("-clusternum", type = int, default = 10)
    parser.add_argument("-clustersize", type = int, default = 1)
    parser.add_argument("-clustercommunicationtime", type = int, default = 0)
    # how to choose epoch for each client, cluster, centralserver
    parser.add_argument("-centralserverepoch", type = int, default = 100)
    parser.add_argument("-clusterepochtype", type = str, choices = ["fixed", "custom"], default = "fixed")
    parser.add_argument("-clusterepoch", type = int, default = 1)
    parser.add_argument("-localepochtype", type =str, choices=["fixed", "custom"], default ="fixed")
    parser.add_argument("-localepoch", type = int, default = 1)
    # details
    parser.add_argument("-intraasyncalpha", type = float, default = 0.6)
    parser.add_argument("-intraasyncthreshold", type = int, default = 5)
    parser.add_argument("-interasyncalpha", type = float, default = 0.6)
    parser.add_argument("-interasyncthreshold", type = int, default = 5)
    parser.add_argument("-optimizername", type = str, default = "sgd")
    parser.add_argument("-learningrate", type = float, default = 0.01)
    parser.add_argument("-batchsize", type = int, default =32)
    parser.add_argument("-randomseed", type = int, default = 1)
    parser.add_argument("-device", type = str, default = "cuda")
    parser.add_argument("-dominantpercentage", type = int, default = 95)
    parser.add_argument("-labelperclient", type = int, default =2)
    parser.add_argument("-dirichletalpha", type = float, default = 0.1)
    parser.add_argument("-regularizationcoefficient", type =float, default = 0.0)
    args = parser.parse_args()

    # set logger for the general training process
    # setting for the multi proecessing in intracluster-sync setting
    # set args.testdataloader for the validation across global distribution
    # set args.traindataset for training data
    # set args.labellist for the label list
    setting(args)
    
    # set communication and system time for each client
    # split the data and distribute to each clients as a list. and also save the data distribution for each client in args.labelpercentageperclient, and global data distribution in args.labelpercentageforglobaldistribution
    # create list of clients in this stage, local epochs, clientid, clusterid for clients are not set up yet
    clientcommunicationtimelist, clientcomputationtimelist = partition_system(args)
    clientdataloaderlist = partition_data(args)
    clientlist = create_clientlist(clientdataloaderlist, clientcommunicationtimelist, clientcomputationtimelist, args)

    # clustering 
    # set central server's centralserverepoch, interaggregation strategy
    # set cluster's clusterid, communicationtime, intraaggregation strategy, clusterepoch
    # set client's clusterid, clientid, local epoch
    centralserver = clusterclients(clientlist, args)

    # log experiment setting(args)
    # log client information
    # set loggers for each cluster, log the cluster information
    logsetting(args)
    logclient(clientlist,args)
    logcluster(centralserver,args)

    # training process
    centralserver.central_train()

    # save graph file
    savegraph(args)

