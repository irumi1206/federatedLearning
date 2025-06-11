from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import os
import random
from datasets import load_dataset

# Return list of dataloader splited based on the setting also save data distribution for each client and global data distribution(args.labelpercentageperclient, args.labelpercentageforglobaldistribution)
def partition_data(args):

    if args.datasetname == "femnist":
        print("d")



    # return list of dataloader based on the splitting type
    if args.dataheterogeneitytype == "iid":
        dataloaderlist = split_iid(args)
    elif args.dataheterogeneitytype == "onelabeldominant":
        dataloaderlist = split_onelabeldominant(args)
    elif args.dataheterogeneitytype == "onlyspecificlabel":
        dataloaderlist = split_onlyspecificlabelexist(args)
    elif args.dataheterogeneitytype == "dirichletdistribution":
        dataloaderlist = split_dirichletdistribution(args)
    else:
        raise ValueError("not supported data split")
    
    # error checking the partition is done in correct number
    if len(dataloaderlist) != args.clientnum:
        raise ValueError("partitioning error")

    # save global label percentage
    args.labelpercentageforglobaldistribution = defaultdict(float)
    traindasetcache=[]
    for _,label in args.traindataset:
        traindasetcache.append(label)
    for label in args.labellist:
        args.labelpercentageforglobaldistribution[label] = (len([i for i in traindasetcache if i == label])/len(traindasetcache))

    return dataloaderlist

# Return list of dataloader splitted in iid manner with similar size
def split_iid(args):

    # get parameter
    totalsample = len(args.traindataset)

    # shuffle the dataset and split it
    shuffledindices = np.random.permutation(totalsample)
    clientindices = np.array_split(shuffledindices, args.clientnum)
    dataloaderlist = [DataLoader(Subset(args.traindataset, indices), batch_size = args.batchsize, shuffle = True) for indices in clientindices]
    
    return dataloaderlist

# Return list of dataloader splitted in clients having one dominant label
# Client with similar dominant label is close to each other
def split_onelabeldominant(args):

    # get parameter and setting
    ratio = args.dominantpercentage
    clientnum = args.clientnum
    labeltoindices = defaultdict(list)
    for i in range(len(args.traindataset)):
        _, label = args.traindataset[i]
        labeltoindices[label].append(i)
    labellist = [label for label in labeltoindices.keys()]
    if len(labellist) > args.clientnum : raise NotImplementedError("too small clients")
    rng = np.random.default_rng(args.randomseed+4)

    # compute client number per label, fair distribution, labels with few get less clients
    sortedlabellist = sorted(labellist, key = lambda x : len(labeltoindices[x]),reverse = True)
    labeltodevicenum = defaultdict(int)
    for i in range(len(sortedlabellist)):
        if i<(clientnum % len(labellist)):
            labeltodevicenum[sortedlabellist[i]] = int(clientnum / len(labellist))+1
        else:
            labeltodevicenum[sortedlabellist[i]] = int(clientnum / len(labellist))

    # split indices used for dominance
    nondominantlabeltoindices = defaultdict(list)
    dominantlabeltoindices = defaultdict(list)
    for label,indicelist in labeltoindices.items():
        splitpoint = int(len(indicelist)*ratio/100.0)
        if ratio !=100: nondominantindicessplit = np.array_split(indicelist[splitpoint:],len(labeltoindices.keys())-1)
        else: 
            nondominantindicessplit=[[] for _ in range(len(labeltoindices) - 1)]
        current = 0
        for i in labeltoindices.keys():
            if label != i:
                nondominantlabeltoindices[i].extend(nondominantindicessplit[current])
                current += 1
        dominantlabeltoindices[label] = indicelist[:splitpoint]

    # for each label, extract dominant data from labeltoindices and rest from the nondominantindices excluding the label
    dataloaderlist = []
    for label,indiceslist in dominantlabeltoindices.items():
        devicenum = labeltodevicenum[label]
        rng.shuffle(indiceslist)
        dominantindices = np.array_split(indiceslist,devicenum)
        rng.shuffle(nondominantlabeltoindices[label])
        if ratio !=100 : 
            nondominantindices = np.array_split(nondominantlabeltoindices[label],devicenum)
        else: 
            nondominantindices = [[] for _ in range(devicenum)]
        for i in range(devicenum):
            if ratio !=100: dataloaderlist.append(DataLoader(Subset(args.traindataset, np.concatenate((dominantindices[i],nondominantindices[i]))), batch_size = args.batchsize, shuffle=True))
            else: dataloaderlist.append(DataLoader(Subset(args.traindataset, dominantindices[i]), batch_size = args.batchsize, shuffle=True))
    return dataloaderlist

# Return list of dataloader splitted in clients having only two labels
# Randomly choose labels for each client, and distribute data to each client, the dominant label is totaly random
def split_onlyspecificlabelexist(args):

    # get parameter and setting
    labeltoindices = defaultdict(list)
    for i in range(len(args.traindataset)):
        _, label = args.traindataset[i]
        labeltoindices[label].append(i)
    labellist = [label for label in labeltoindices.keys()]
    rng = np.random.default_rng(args.randomseed+4)

    # setting for label partitioning to clients
    labelchosennum = defaultdict(int)
    labelchosendevice = defaultdict(list)
    for label in labellist:
        labelchosennum[label] =0
        labelchosendevice[label] = []

    # set label for each client
    for i in range(args.clientnum):

        # choose labels for each client
        chosenlabel = rng.choice(labellist, size = args.labelperclient, replace = False)

        # record the chosen labels per label
        for label in chosenlabel:
            labelchosennum[label] += 1
            labelchosendevice[label].append(i)

    # distribute data to each client
    indiceperclient = defaultdict(list)
    for i in range(args.clientnum):
        indiceperclient[i] = []

    for label in labellist:
        if labelchosennum[label] > 0:
            rng.shuffle(labeltoindices[label])
            indiceslist = np.array_split(labeltoindices[label],labelchosennum[label])
            for i in range(labelchosennum[label]):
                indiceperclient[labelchosendevice[label][i]].extend(indiceslist[i])

    # create dataloader for each client
    dataloaderlist = []
    for i in range(args.clientnum):
        dataloaderlist.append(DataLoader(Subset(args.traindataset, indiceperclient[i]), batch_size = args.batchsize, shuffle = True))

    return dataloaderlist
        
# Return list of dataloader splitted in clients following the dirichlet distribution
# Dominant label is random, and also the dataset size is random, some clients might have almost no data
def split_dirichletdistribution(args):

    # get parameter
    alpha = args.dirichletalpha
    rng = np.random.default_rng(args.randomseed+4)
    clientnum = args.clientnum

    # get labels
    labels = np.array(args.traindataset.targets)
    uniquelabels = np.unique(labels)
    clientindices = defaultdict(list)

    # split data to each client
    for label in uniquelabels:
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet([alpha] * clientnum)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_idxs = np.split(indices, proportions)
        for clientid, idx in enumerate(split_idxs):
            clientindices[clientid].extend(idx.tolist())

    # create dataloader for each client
    dataloaderlist = []
    for i in range(clientnum):
        subset = Subset(args.traindataset, clientindices[i])
        loader = DataLoader(subset, batch_size=args.batchsize, shuffle=True)
        dataloaderlist.append(loader)

    return dataloaderlist


