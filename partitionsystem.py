import numpy as np

def partition_system(args):

    rng = np.random.default_rng(args.randomseed+5)

    # Systen configuration
    networkvariance = 0.0
    devcievaricnae = 0.0
    basecomputationtime =0.1 # base time per epoch
    basecommunicationtime = 0.05 # base time per model transfer

    # Device capability category
    devicecpapbility={
        'high': 100,
        'medium': 200,
        'low': 400
    }
    devicedistribution = [0.2, 0.5, 0.3]

    # Network condition category
    networkcondition={
        'high': 100,
        'medium': 200,
        'low': 400
    }
    networkdistribution = [0.3, 0.4, 0.3]

    clustercommunicationtimelist = []
    clientcommunicationtimelist = []
    clientcomputationtimelist = []

    # Assign communication and computation time to each cluster and client
    for i in range(args.clusternum):
        networkmultiplier = rng.choice(list(networkcondition.values()), p=networkdistribution) * (1 + rng.uniform(-networkvariance, networkvariance))
        clustercommunicationtimelist.append(int(basecommunicationtime * networkmultiplier))
    
    for i in range(args.clientnum):
        networkmultiplier = rng.choice(list(networkcondition.values()), p=networkdistribution) * (1 + rng.uniform(-networkvariance, networkvariance))
        computationmultiplier = rng.choice(list(devicecpapbility.values()), p=devicedistribution) * (1 + rng.uniform(-devcievaricnae, devcievaricnae))
        clientcommunicationtimelist.append(int(basecommunicationtime * networkmultiplier))
        clientcomputationtimelist.append(int(basecomputationtime * computationmultiplier))

    # Error checking the partition is done in correct number
    if len(clientcommunicationtimelist) != args.clientnum or len(clientcomputationtimelist) != args.clientnum or len(clustercommunicationtimelist) != args.clusternum:
        raise ValueError("partitioning error")

    return clustercommunicationtimelist, clientcommunicationtimelist, clientcomputationtimelist