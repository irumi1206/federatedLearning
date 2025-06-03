# federatedLearning

Implementation about federated learning and boosting its convergence via system and network modeling. Various simulations can be done via configuration


# specification about the parameter

-modelname : Name of the model for training. choices are

    cnnmnist : cnn model for mnist

    cnncifar10 : cnn model for cifar10

    Additional models can be added by adding the modelname to the choices in paser, addding the structure of the model and the get_model function as a file to the models folder, and specifying the model at the get_model function at utils.py

-datasetname : Name of the dataset for training and testing. choices are

    mnist : mnist dataset

    cifar10 : cifar10 dataset

    Additional dataset can be added by adding the datasetname to the choices in parser, and adding how to get dataset class for training and test in get_test_dataset and get_train_dataset in utils.py

-intraclusteringtype : Aggregation strategy within the cluster. choices are

    sync : adopts synchronous aggregation for intraclustering

    async : adopts asynchronous aggregation for intraclustering with threshold(-intraasyncthreshold) and staleness alpha(-intraasyncalpha)

-interclusteringtype : Aggregation strategy between the clusters. choices are

    sync : adopts synchronous aggregation for interclustering

    async : adopts asynchronous aggregation for interclustering with threshold(-interasyncthreshold) and staleness alpha(-interasyncalpha)

-clientnum : Total number of clients participating in federated learning

-dataheterogeneity : Data distribution for each client. choices are

    onelabeldominant : each device is set certain percentage(-dominantpercentage) of one label and the rest is filled with other labels. When the number of clients does not match exactly to assign equal number of clients for each label, labels with more dataset is assinged more clients for dominant label. After split, devices with same dominant label is placed right next to each other.
    
    onlyspecificlabel : each device only has specific number of labels(-labelperclient). Each device chooses certain labels to hold at first, then data is distributed. So there is no guarantee that the label distribution for each device is evenly selected, making the size of data in device totally random and there is no correlation in dataset between devices next to each other.
    
    dirichletdistribution : the data is splitted following the dirichlet distribution with alpha(-dirichletalpha) where the heterogeneity can be controlled. Same as the onlyspecificlabelpartition, there is no consistency for the distribution of each label and no correlation between devices next to each other.

-systemheterogeneity : System distribution for each client. cluster - central communication time is set to 500msec, for clients the choices are

    alltimesame : all the computation time and communication time is set to 100msec

    communicationtimesamecomputationdifferent : all the communication time set to 100msec, and the computation is chosen from [100,200,300,400,500] with same probabilities

    realistic : the communication and computation of clients is dependent on the class of device it belongs to. 5 classes include highend laptop, mid range laptop/high end tablet, fast smartphone, mid range smartphone/tablet, low end smartphone/rasberry pi. each device is selected as one of the class by the distribution of devices

    custom : for need to set communication and computation time manually

-clusteringtype : how to cluster clients. choices are

    clusterbyclientorder : given a list of clients, cluster in client order with fixed clusternum(-clustersize) and clustersize(-clustersize)

    clusterbyrandomshuffle : randomly cluster clients with fixed clusternum(-clusternum) and clustersize(-clustersize)

    custom : for need to set clustering manually

-centralserverepoch : how many rounds to train in centralserver

-clusterepochtype : how to choose the round for clusters before aggregation to the central server. choices are

    fixed : all the clusters have fixed clusterepoch(-clusterepoch)

    custom : for need to set clusterepoch manually after clustering

-localepochtype : how to choose the epoch for clients before aggregation to the cluster head. choices are

    fixed : all the clients have fixed localepoch(-localepoch)

    custom : for need to set localepoch manually after clustering

-optimizername : optimzer name to use as optimizer. choices are

    sgd : sgd optimizer

    additional optimizer can be added by adding it to parser choices, and adding the optimzer to the get_optimizer in utils.py

-learningrate : learning rate for training

-batchsize : batch size for training

-randomseed : random seed for reproductability

-device : device to use for training at current os. choices are

    cuda : cuda
    