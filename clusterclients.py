from centralserver import CentralServer
from cluster import Cluster
from client import Client
import random

def cluster_clients(clientlist, args):

    centralserver = CentralServer(args.interclusteringtype, args.centralserverepoch, args, [])

    # clustering based on client order, clustersize and number is set
    if args.clusteringtype == "clusterbyclientorder":

        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            for clientind in range(args.clustersize):
                client = clientlist[clusterind*args.clustersize + clientind]
                client.clientid = clientind
                client.clusterid = clusterind
                cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)
        
        return centralserver

    # random shuffle the clients once, then cluster by client order
    elif args.clusteringtype == "clusterbyrandomshuffle":

        shuffledind=[i for i in range(args.clientnum)]
        random.shuffle(shuffledind)

        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            for clientind in range(args.clustersize):
                client = clientlist[shuffledind[clusterind*args.clustersize + clientind]]
                client.clientid = clientind
                client.clusterid = clusterind
                cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)

        return centralserver
    
    elif args.clusteringtype == "clusterbygradientsimilarity":

        raise ValueError("to be implemented")
        
    
    elif args.clusteringtype == "custom":

        raise ValueError("customize it plz")
    
    else:

        raise ValueError("clustering type not supported")


        