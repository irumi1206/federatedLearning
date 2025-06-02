from centralserver import CentralServer
from cluster import Cluster
from client import Client

def clusterclients(clientlist, clustercommunicationtimelist, args):

    clusteredclientlist = [[]for _ in range(args.clusternum)]

    # clustering
    for clusterind in range(args.clusternum):
        for clientind in range(args.clustersize):
            clusteredclientlist[clusterind].append(clientlist[clusterind*args.clustersize + clientind])
    
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


    centralserver = CentralServer(args,[
        Cluster(i,clustercommunicationtimelist[i],args,[clientlist[i*args.clustersize+j] 
            for j in range(args.clustersize)
        ])
        for i in range(args.clusternum) 
    ])

    return centralserver


    # clustering
    clusterlist =[[] for _ in range(args.clusternum)]
    
    for clusterind in range(args.clusternum):
        for clientind in range(args.clustersize):
            clusterlist[clusterind].append(clientlist[clusterind*args.clustersize+clientind])
    
    # organise cluster and central server
    centralserver = CentralServer(args,[])

    for clusterind in range(args.clusternum):

        cluster = Cluster(clusterind, clustercommunicationlist[clusterind], args, [])

        for clientind in range(args.clustersize):

            client = clusterlist[clusterind][clientind]
            client.clientid = clientind
            client.clusterid = clusterind
            cluster.clientlist.append(client)

        centralserver.clusterlist.append(cluster)

    return centralserver
        