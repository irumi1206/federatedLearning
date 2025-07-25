import numpy as np

def partition_system(args):

    # fix randomness
    rng = np.random.default_rng(args.settingrandomseed+5)

    # cluster communication time, client communicationtime, and computation time is all set to 100mse
    if args.systemheterogeneity == "alltimesame":
        clientcommunicationtimelist = [100 for _ in range(args.clientnum)]
        clientcomputationtimelist = [100 for _ in range(args.clientnum)]

    # cluster communication time and client communication time is set to 100msec, and computatation is uniformed by distribution of [100.200.300.400.500]
    elif args.systemheterogeneity == "communicationtimesamecomputationdifferent":
        clientcommunicationtimelist = [100 for _ in range(args.clientnum)]
        clientcomputationtimelist = []

        for _ in range(args.clientnum):
            clientcomputationtimelist.append(int(rng.choice([100, 200, 300, 400, 500], p=[0.2, 0.2, 0.2, 0.2, 0.2])))

    elif args.systemheterogeneity == "realistic":

        # variance for computation and communication time
        networkvariance = 0.2
        computationvariance = 0.2

        # assumption for device class's computation and communication time
        # assuming the model size is 10mb and the batchsize is 32, the training time per batch and the upload communication time is calculated
        # assume upload time is similar to download
        # the ratio of each device's capabilities and the ratio of the communication time and computation time is the key, so even with larger modeel size,
        # since the communication and computation time ratio would be about the same, the value is enough to examine the fast convergence in relavance to time

        # communication time for device (msec)
        devicecommunicationtime={
            "highend recent smartphones(top tier 5g/wifi)" : 800,
            "midrange recent smartphones(typical 4g/mid range 5g)" : 2400,
            "low range recent smartphones(slow 4g)" : 24000,
            "old smartphones(slow 4g)" : 24000,
            "iot devices(lora)" : 2400000
        }

        #Device trainigtime per batch (msec)
        devicetrainingtime={
            "highend recent smartphones(gpubounded)" : 10,
            "midrange recent smartphones(gpubounded)" : 25,
            "low range recent smartphones(gpubounded)" : 70,
            "old smartphones(cpubounded))" : 600,
            "iot devices(cpubounded))" : 5000
        }

        # distribution for each device
        devicedistribution = [0.10, 0.35, 0.30, 0.20, 0.05]

        clientcommunicationtimelist = []
        clientcomputationtimelist = []
        
        # assign communication and computation time for each client
        for i in range(args.clientnum):
            clientdevice = rng.choice([0,1,2,3,4], p=devicedistribution)
            communicationtime = list(devicecommunicationtime.values())[clientdevice]
            computationtime = list(devicetrainingtime.values())[clientdevice]
            clientcommunicationtimelist.append(int(communicationtime * (1 + rng.uniform(-networkvariance, networkvariance))))
            clientcomputationtimelist.append(int(computationtime  * (1 + rng.uniform(-computationvariance, computationvariance))))

    elif args.systemheterogeneity == "comp100200random_comm100":

        ######################################
        clientcommunicationtimelist = [100 for _ in range(args.clientnum)]
        clientcomputationtimelist = []
        for i in range(args.clientnum):
            time = int(rng.choice([100,200], p=[0.5,0.5]))
            clientcomputationtimelist.append(time)

    elif args.systemheterogeneity == "assume100clients95onelabeldominance_foreachlabel100200300400500of2randomlydistributed_comm3":

        clientcomputationtimelist = []
        distributionforlabelgroup = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500]
        for _ in range(10):
            copied = distributionforlabelgroup.copy()
            rng.shuffle(copied)  # In-place shuffle
            clientcomputationtimelist.extend(copied)

        clientcommunicationtimelist = []
        for i in range(args.clientnum):
            clientcommunicationtimelist.append(clientcomputationtimelist[i]*3)

    elif args.systemheterogeneity == "uniformdistribution":

        clientcomputationtimelist = []
        clientcommunicationtimelist = []
        
        for i in range(args.clientnum):
            clientcomputationtimelist.append(int(rng.uniform(low=100.0, high=500.0)))
        for i in range(args.clientnum):
            clientcommunicationtimelist.append(clientcomputationtimelist[i]*2)

    elif args.systemheterogeneity == "uniformdistribution2":

        computationvariance = 0.2

        #Device trainigtime per batch (msec)
        devicetrainingtime={
            "highend recent smartphones(gpubounded)" : 100,
            "midrange recent smartphones(gpubounded)" : 200,
            "low range recent smartphones(gpubounded)" : 300,
            "old smartphones(cpubounded))" : 400,
            "iot devices(cpubounded))" : 500
        }

        # distribution for each device
        devicedistribution = [0.2,0.2,0.2,0.2,0.2]

        clientcommunicationtimelist = []
        clientcomputationtimelist = []
        
        # assign communication and computation time for each client
        for i in range(args.clientnum):
            clientdevice = rng.choice([0,1,2,3,4], p=devicedistribution)
            computationtime = list(devicetrainingtime.values())[clientdevice]
            time = int(computationtime * (1 + rng.uniform(-computationvariance, computationvariance)))

            clientcomputationtimelist.append(time)
            clientcommunicationtimelist.append(time*2)
   
    else:
        raise ValueError("")

    # Error checking the partition is done in correct number
    if len(clientcommunicationtimelist) != args.clientnum or len(clientcomputationtimelist) != args.clientnum:
        raise ValueError("partitioning error")

    return clientcommunicationtimelist, clientcomputationtimelist