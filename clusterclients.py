from centralserver import CentralServer
from cluster import Cluster
from client import Client

def clusterclients(clientlist, args):

    clusteredclientlist = [[]for _ in range(args.clusternum)]

    # clustering
    if args.clusteringtype == "clusterbyclientorder":
        for clusterind in range(args.clusternum):
            for clientind in range(args.clustersize):
                clusteredclientlist[clusterind].append(clientlist[clusterind*args.clustersize + clientind])
    elif args.clusteringtype == "clusterbyrandomshuffle":
        raise ValueError("not implemented yet")
    elif args.clusteringtype == "custom":
        raise ValueError("customize it plz")
    else:
        raise ValueError("clustering type not supported")


    # organizing central server and clusters and set cluster-central communication time(500msec)
    tempclusterlist = []
    for clusterind in range(args.clusternum):
        tempclientlist = []
        for clientind in range(args.clustersize):
            client = clusteredclientlist[clusterind][clientind]
            client.clientid = clientind
            client.clusterid = clusterind
            tempclient = Client(clusterind, clientind, client.dataloader, client.communicationtime, client.computationtimeperbatch, client.uniqueid,1, args)
            tempclientlist.append(tempclient)
        cluster = Cluster(clusterind, 500, args, tempclientlist)
        tempclusterlist.append(cluster)
    centralserver = CentralServer(args, tempclusterlist)



    return centralserver

        