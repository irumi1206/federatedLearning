from centralserver import CentralServer
from cluster import Cluster
from client import Client

def clusterclients(clientlist, clustercommunicationtimelist, args):

    clusteredclientlist = [[]for _ in range(args.clusternum)]

    # clustering
    if args.clusteringtype == "clusterbypartitionorder":
        for clusterind in range(args.clusternum):
            for clientind in range(args.clustersize):
                clusteredclientlist[clusterind].append(clientlist[clusterind*args.clustersize + clientind])
    elif args.clusteringtype == "clusterbycustom":
        for clusterind in range(args.clusternum):
            for clientind in range(args.clustersize):
                clusteredclientlist[clusterind].append(clientlist[clientind*args.clustersize + clusterind])
    else:
        raise ValueError("wrong clustering type")


    # organizing central server and clusters
    tempclusterlist = []
    for clusterind in range(args.clusternum):
        tempclientlist = []
        for clientind in range(args.clustersize):
            client = clusteredclientlist[clusterind][clientind]
            client.clientid = clientind
            client.clusterid = clusterind
            tempclient = Client(clusterind, clientind, client.dataloader, client.communicationtime, client.computationtime, client.uniqueid, args)
            tempclientlist.append(tempclient)
        cluster = Cluster(clusterind, clustercommunicationtimelist[clusterind], args, tempclientlist)
        tempclusterlist.append(cluster)
    centralserver = CentralServer(args, tempclusterlist)

    return centralserver

        