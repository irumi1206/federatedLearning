import numpy as np

def partition_system(args):

    # fix randomness
    rng = np.random.default_rng(args.randomseed+5)

    # cluster communication time, client communicationtime, and computation time is all set to 100mse
    if args.systemheterogeneity == "alltimesame":
        clientcommunicationtimelist = [100 for _ in range(args.clientnum)]
        clientcomputationtimelist = [100 for _ in range(args.clientnum)]

    # cluster communication time and client communication time is set to 100msec, and computatation is uniformed by distribution of [100.200.300.400.500]
    elif args.systemheterogeneity == "communicationtimesamecomputationdifferent":
        clientcommunicationtimelist = [100 for _ in range(args.clientnum)]
        clientcomputationtimelist = []

        for _ in range(args.clientnum):
            clientcomputationtimelist.append(rng.choice([100, 200, 300, 400, 500], p=[0.2, 0.2, 0.2, 0.2, 0.2]))

    elif args.systemheterogeneity == "realistic":

        # variance for computation and communication time
        networkvariance = 0.2
        computationvariance = 0.2

        # assumption for device class's computation and communication time
        # assuming the model size is 2mb and the batchsize is 32, the training time per batch and the upload communication time is calculated
        # assume upload time is similar to download
        # the ratio of each device's capabilities and the ratio of the communication time and computation time is the key, so even with larger modeel size,
        # since the communication and computation time ratio would be about the same, the value is enough to examine the fast convergence in relavance to time

        # communication time for device
        devicecommunicationtime={
            "highend laptop" : 50.0,
            "mid range laptop/high end tablet" : 150.0,
            "fast smartphone" : 300.0,
            "mid range smartphone/tablet" : 1000.0,
            "low end smartphone/rasberry pi" : 5000.0
        }

        # Device trainigtime per batch (msec)
        devicetrainingtime={
            "highend laptop" : 100.0,
            "mid range laptop/high end tablet" : 200.0,
            "fast smartphone" : 400.0,
            "mid range smartphone/tablet" : 800.0,
            "low end smartphone/rasberry pi" : 1500.0
        }

        # distribution for each device
        devicedistribution = [0.1, 0.2, 0.2, 0.4, 0.1]

        clustercommunicationtimelist = []
        clientcommunicationtimelist = []
        clientcomputationtimelist = []
        
        # assign communication and computation time for each client
        for i in range(args.clientnum):
            clientdevice = rng.choice([0,1,2,3,4], p=devicedistribution)
            communicationtime = list(devicecommunicationtime.values())[clientdevice]
            computationtime = list(devicetrainingtime.values())[clientdevice]
            clientcommunicationtimelist.append(int(communicationtime * (1 + rng.uniform(-networkvariance, networkvariance))))
            clientcomputationtimelist.append(int(computationtime  * (1 + rng.uniform(-computationvariance, computationvariance))))

    elif args.systemheterogeneity == "custom":

        ######################################



        #######################################
        raise ValueError("customize it plz")
        clientcommunicationtimelist = [0,0,0,0,0,0,0,0,0,0]
        clientcomputationtimelist = [100,100,100,100,100,50,50,50,50,50]

    else:
        raise ValueError("")

    # Error checking the partition is done in correct number
    if len(clientcommunicationtimelist) != args.clientnum or len(clientcomputationtimelist) != args.clientnum:
        raise ValueError("partitioning error")

    return clientcommunicationtimelist, clientcomputationtimelist