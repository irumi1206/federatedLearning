from centralserver import CentralServer
from cluster import Cluster
from client import Client
import random

def clusterclients(clientlist, args):

    clusterlist = []

    # clustering
    if args.clusteringtype == "clusterbyclientorder":

        for clusterind in range(args.clusternum):
            tempcluster = []
            for clientind in range(args.clustersize):
                tempcluster.append(clientlist[clusterind*args.clustersize + clientind])
            clusterlist.append(tempcluster)

    elif args.clusteringtype == "clusterbyrandomshuffle":

        clientind=[i for i in range(args.clientnum)]
        random.shuffle(clientind)

        for clusterind in range(args.clusternum):
            tempcluster = []
            for clientind in range(args.clustersize):
                tempcluster.append(clientlist[clientind[clusterind*args.clustersize + clientind]])
            clusterlist.append(tempcluster)

    
    elif args.clusteringtype == "custom":

        for clusterind in range(args.clusternum):
            tempcluster = []
            for clientind in range(args.clustersize):
                tempcluster.append(clientlist[clusterind*args.clustersize + clientind])
            clusterlist.append(tempcluster)

        #raise ValueError("customize it plz")
    
    else:

        raise ValueError("clustering type not supported")
    
    # organize the clusterlist into trees of root with centralserver
    # set centralserver's interclusteringtype, centralserverepoch
    # set cluster's clusterid, communicationtime, intraclusteringtype, clusterepoch
    # set client's clusterid, clientid, localepoch
    centralserver = CentralServer(args.interclusteringtype, args.centralserverepoch, args,[])

    for clusterind in range(len(clusterlist)):

        if args.clusterepochtype == "fixed": cluster = Cluster(clusterind, 500 if args.systemheterogeneity == "realistic" else args.clustercommunicationtime , args.intraclusteringtype, args.clusterepoch, args, [])
        elif args.clusterepochtype == "custom": raise ValueError("not implemented yet")
        else: raise ValueError("clusterepochtype not supported")

        for clientind in range(len(clusterlist[clusterind])):

            client = clusterlist[clusterind][clientind]
            client.clusterid = clusterind
            client.clientid = clientind

            if args.localepochtype == "fixed": client.localepoch = args.localepoch
            elif args.localepochtype =="custom": 
                if clusterind%2 == 0: client.localepoch = args.localepoch*2
                else: client.localepoch = args.localepoch
                #raise ValueError("not implemented yet")
            else: raise ValueError("localepochtype not supported")

            cluster.clientlist.append(client)
        
        centralserver.clusterlist.append(cluster)

    return centralserver





    # # organizing central server and clusters and set cluster-central communication time(500msec)
    # tempclusterlist = []
    # for clusterind in range(args.clusternum):
    #     tempclientlist = []
    #     for clientind in range(args.clustersize):
    #         client = clusterlist[clusterind][clientind]
    #         if args.localepochtype == "fixed" :tempclient = Client(clusterind, clientind, client.dataloader, client.communicationtime, client.computationtimeperbatch, client.uniqueid,args.localepoch, args)
    #         elif args.localepochtype == "dynamic" : raise ValueError("not implemented yet")
    #         else : raise ValueError("localepochtype not supported")
    #         tempclientlist.append(tempclient)
    #     if args.clusterepochtype == "fixed" : cluster = Cluster(clusterind, 500, args.intraclusteringtype, args.clusterepoch, args.clustersize, args, tempclientlist)   
    #     elif args.clusterepochtype == "dynamic:" : raise ValueError("not implemented yet")
    #     else : raise ValueError("clusterepochtype not supported")
    #     tempclusterlist.append(cluster)
    # centralserver = CentralServer(args, tempclusterlist)



    return centralserver

        