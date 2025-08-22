import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

#from dataset.FEMNISTClientDataset import FEMNISTClientDataset


# Return list of dataloader splited based on the setting also save data distribution for each client and global data distribution(args.labelpercentageperclient, args.labelpercentageforglobaldistribution)
def partition_data(args):

    if args.datasetname == "femnist":

        dataloaderlist = []
        args.labelpercentageforglobaldistribution = defaultdict(float)
        dataperlabel = defaultdict(int)
        dataperclient = defaultdict(list)
        totalsample = len(args.traindataset)
        
        for sample in args.traindataset:
            dataperlabel[sample["character"]] +=1
            dataperclient[sample["writer_id"]].append(sample)

        for label in dataperlabel.keys():
            args.labelpercentageforglobaldistribution[label] = dataperlabel[label]/totalsample
        
        for samples in dataperclient.values():
            clientdataset = FEMNISTClientDataset(samples)
            clientdataloader = DataLoader(clientdataset, batch_size=args.batchsize, shuffle =True)
            dataloaderlist.append(clientdataloader)


        if len(dataloaderlist) != args.clientnum:
            raise ValueError("partitioning error")

        return dataloaderlist
            

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
    rng = np.random.default_rng(args.settingrandomseed)
    shuffledindices = rng.permutation(totalsample)
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
    rng = np.random.default_rng(args.settingrandomseed)

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
    rng = np.random.default_rng(args.settingrandomseed)

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


# def split_dirichletdistribution(args):
#     """
#     Splits data using a client-skewed Dirichlet distribution, ensuring
#     each client has an equal total number of data samples.
#     """
#     # 1. Get parameters
#     alpha = args.dirichletalpha
#     rng = np.random.default_rng(args.settingrandomseed)
#     num_clients = args.clientnum
    
#     # 2. Get labels and data indices sorted by label
#     labels = np.array(args.traindataset.targets)
#     num_classes = len(np.unique(labels))
#     indices_by_label = [np.where(labels == i)[0] for i in range(num_classes)]
    
#     # Shuffle the indices within each label pool for random drawing
#     for idx_list in indices_by_label:
#         rng.shuffle(idx_list)
    
#     # 3. Determine the fixed number of data points per client
#     total_data_size = len(labels)
#     data_per_client = total_data_size // num_clients
    
#     # 4. For each client, generate its unique "recipe" of label proportions
#     client_proportions = rng.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    
#     client_indices = [[] for _ in range(num_clients)]
    
#     # Pointers to track the current position in each label's index pool
#     label_pool_pointers = np.zeros(num_classes, dtype=int)
    
#     # 5. For each client, "draw" data from the pools according to its recipe
#     for client_id in range(num_clients):
#         proportions = client_proportions[client_id]
        
#         # Calculate how many samples of each label this client should get
#         samples_per_label = (proportions * data_per_client).astype(int)
        
#         # Correct for rounding errors to ensure client gets exactly data_per_client samples
#         remainder = data_per_client - samples_per_label.sum()
#         samples_per_label[np.argmax(proportions)] += remainder
        
#         # Draw the required number of samples for each label
#         for label_idx, num_samples in enumerate(samples_per_label):
#             # Check if there are enough samples left in the pool
#             if label_pool_pointers[label_idx] + num_samples > len(indices_by_label[label_idx]):
#                 num_samples = len(indices_by_label[label_idx]) - label_pool_pointers[label_idx]

#             start_pos = label_pool_pointers[label_idx]
#             end_pos = start_pos + num_samples
            
#             client_indices[client_id].extend(indices_by_label[label_idx][start_pos:end_pos])
            
#             # Move the pointer for the next client
#             label_pool_pointers[label_idx] = end_pos

#     # 6. Create a dataloader for each client
#     dataloaderlist = []
#     for i in range(num_clients):
#         subset = Subset(args.traindataset, client_indices[i])
#         loader = DataLoader(subset, batch_size=args.batchsize, shuffle=True)
#         dataloaderlist.append(loader)

#     return dataloaderlist



        
# Return list of dataloader splitted in clients following the dirichlet distribution
# Dominant label is random, and also the dataset size is random, some clients might have almost no data
# def split_dirichletdistribution(args):

#     # get parameter
#     alpha = args.dirichletalpha
#     rng = np.random.default_rng(args.settingrandomseed)
#     clientnum = args.clientnum

#     # get labels
#     labels = np.array(args.traindataset.targets)
#     uniquelabels = np.unique(labels)
#     clientindices = defaultdict(list)

#     # split data to each client
#     for label in uniquelabels:
#         indices = np.where(labels == label)[0]
#         rng.shuffle(indices)
#         proportions = rng.dirichlet([alpha] * clientnum)
#         proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
#         split_idxs = np.split(indices, proportions)
#         for clientid, idx in enumerate(split_idxs):
#             clientindices[clientid].extend(idx.tolist())

#     # create dataloader for each client
#     dataloaderlist = []
#     for i in range(clientnum):
#         subset = Subset(args.traindataset, clientindices[i])
#         loader = DataLoader(subset, batch_size=args.batchsize, shuffle=True)
#         dataloaderlist.append(loader)

#     return dataloaderlist

def split_dirichletdistribution(args):
    """
    Splits data among clients using a client-skewed Dirichlet distribution.
    Each client receives a unique distribution of labels.
    """
    # get parameters
    alpha = args.dirichletalpha
    rng = np.random.default_rng(args.settingrandomseed)
    clientnum = args.clientnum
    
    # get labels and sort all data indices by their label
    labels = np.array(args.traindataset.targets)
    uniquelabels, label_counts = np.unique(labels, return_counts=True)
    num_labels = len(uniquelabels)
    indices_by_label = [np.where(labels == i)[0] for i in uniquelabels]
    
    # --- Main Logic for Client-Skewed Partitioning ---
    
    # 1. For each client, generate a probability distribution over the labels
    client_proportions = rng.dirichlet([alpha] * num_labels, size=clientnum)
    
    # 2. Balance the client proportions to ensure all data is used
    #    This prevents some labels from being over-assigned while others are under-assigned
    #    by scaling each client's proportion for a label by the total number of clients
    #    that will be assigned that label.
    client_proportions = client_proportions / client_proportions.sum(axis=0)
    
    clientindices = [[] for _ in range(clientnum)]
    
    # 3. For each label, distribute its data indices to clients based on the generated proportions
    for label_idx, indices in enumerate(indices_by_label):
        proportions_for_label = client_proportions[:, label_idx]
        
        # Calculate how many samples of this label each client gets
        samples_per_client = (proportions_for_label * len(indices)).astype(int)
        
        # Correct for rounding errors by assigning remainder to the client with the largest proportion
        remainder = len(indices) - samples_per_client.sum()
        samples_per_client[np.argmax(proportions_for_label)] += remainder
        
        # Split the label's data and assign to clients
        current_pos = 0
        for client_id in range(clientnum):
            num_samples = samples_per_client[client_id]
            client_indices_for_label = indices[current_pos : current_pos + num_samples]
            clientindices[client_id].extend(client_indices_for_label)
            current_pos += num_samples

    # create dataloader for each client
    dataloaderlist = []
    for i in range(clientnum):
        # It's possible a client gets 0 samples in extreme non-IID, handle this
        if not clientindices[i]:
            # Create an empty dataloader or handle as needed
            # For now, we can skip creating a loader for an empty client
            print(f"Warning: Client {i} has no data after partitioning.")
            continue

        subset = Subset(args.traindataset, clientindices[i])
        loader = DataLoader(subset, batch_size=args.batchsize, shuffle=True)
        dataloaderlist.append(loader)

    return dataloaderlist


