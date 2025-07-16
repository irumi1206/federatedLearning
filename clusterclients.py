from centralserver import CentralServer
from cluster import Cluster
from client import Client
import random
from queue import Queue
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np
import tempfile
import shutil
import os

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
    
    #randomly cluster but, to match same cluster level training time, assuming 100clients with onelabel dominant
    elif args.clusteringtype == "clusterbyrandomshufflecustom":

        fixedstraggler = [False] * 100

        for i in range(10):
            for j in range(10):
                if clientlist[i*10+j].computationtime == 500: 
                    fixedstraggler[10*j]=True
                    break
        
        tobeshuffledindices = []
        for i in range(100):
            if not fixedstraggler[i]: tobeshuffledindices.append(i)
        random.shuffle(tobeshuffledindices)

        shuffledind = []
        ind = 0
        for i in range(100):
            if fixedstraggler[i]: shuffledind.append(i)
            else:
                shuffledind.append(tobeshuffledindices[ind])
                ind +=1

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

        # Assuming 'centralserver', 'clientlist', and 'args' are defined

        print("Clustering by gradient similarity...")

        # --- SETUP ---
        # Get the initial model state
        initial_weight_dict = centralserver.model.state_dict()
        with torch.no_grad():
            initial_weight_flat = torch.cat(
                [p.view(-1) for p in initial_weight_dict.values()]
            ).cpu()

        # Define PCA and batch size
        N_COMPONENTS = 20
        BATCH_SIZE = 20  # Must be >= n_components
        pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)

        # Create a temporary directory to store deltas
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for deltas: {temp_dir}")

        try:
            # --- PASS 1: Fit PCA and Save Deltas to Disk ---
            print("Fitting PCA model and caching deltas to disk...")
            delta_batch = []
            client_num = 0
            for client in clientlist:
                client_num += 1
                print(f"Processing Client {client_num}/{len(clientlist)}...")

                # Perform local training
                client.model.load_state_dict(initial_weight_dict)
                client.model.to(args.device)
                q = Queue()
                client.local_train(q)
                with torch.no_grad():
                    client_weight = torch.cat(
                        [p.data.view(-1) for p in client.model.parameters()]
                    ).cpu()
                    delta_weight = initial_weight_flat - client_weight
                
                # Free up GPU memory immediately
                # Also, we no longer need the trained model state in the client object
                client.model.to('cpu')

                # Normalize the delta
                delta_np = delta_weight.numpy()
                row_norm = np.linalg.norm(delta_np)
                normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
                
                # Save the normalized delta to a temporary file
                delta_filepath = os.path.join(temp_dir, f"delta_{client_num - 1}.npy")
                np.save(delta_filepath, normalized_delta)

                # Add the delta to our in-memory batch for fitting
                delta_batch.append(normalized_delta)

                # When the batch is full, fit it and clear the in-memory batch
                if len(delta_batch) == BATCH_SIZE:
                    print(f"  > Fitting PCA with a batch of {len(delta_batch)} deltas...")
                    pca.partial_fit(np.array(delta_batch))
                    delta_batch.clear()

            # Process any remaining deltas from the last batch
            if len(delta_batch) >= N_COMPONENTS:
                print(f"  > Fitting PCA with the final batch of {len(delta_batch)} deltas...")
                pca.partial_fit(np.array(delta_batch))
                delta_batch.clear()

            print("PCA model has been fitted.")

            # --- PASS 2: Load Deltas from Disk and Transform ---
            print("Loading deltas from disk and transforming...")
            reduced_deltas = np.zeros((len(clientlist), N_COMPONENTS))
            for i in range(len(clientlist)):
                # Load the saved delta from disk
                delta_filepath = os.path.join(temp_dir, f"delta_{i}.npy")
                normalized_delta = np.load(delta_filepath).reshape(1, -1)
                
                # Transform the single delta and store it
                reduced_deltas[i] = pca.transform(normalized_delta)
            
            print("All deltas have been transformed.")

            # --- Step 3: Run KMeans ---
            print("Running KMeans on reduced deltas...")
            k = args.clusternum
            kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
            cluster_assignments = kmeans.labels_

            print("Cluster assignments:", cluster_assignments)

        finally:
            # --- CLEANUP ---
            # Always remove the temporary directory and its contents
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            ind = 0
            for clientind in range(len(clientlist)):
                if cluster_assignments[clientind] == clusterind:
                    client = clientlist[clientind]
                    client.clusterid = clusterind
                    client.clientid = ind
                    ind +=1
                    cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)

        return centralserver
    
    elif args.clusteringtype == "clusterbygradientdissimilarity":

        # Assuming 'centralserver', 'clientlist', and 'args' are defined

        print("Clustering by gradient dissimilarity...")

        # --- SETUP ---
        # Get the initial model state
        initial_weight_dict = centralserver.model.state_dict()
        with torch.no_grad():
            initial_weight_flat = torch.cat(
                [p.view(-1) for p in initial_weight_dict.values()]
            ).cpu()

        # Define PCA and batch size
        N_COMPONENTS = 20
        BATCH_SIZE = 20  # Must be >= n_components
        pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)

        # Create a temporary directory to store deltas
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for deltas: {temp_dir}")

        try:
            # --- PASS 1: Fit PCA and Save Deltas to Disk ---
            print("Fitting PCA model and caching deltas to disk...")
            delta_batch = []
            client_num = 0
            for client in clientlist:
                client_num += 1
                print(f"Processing Client {client_num}/{len(clientlist)}...")

                # Perform local training
                client.model.load_state_dict(initial_weight_dict)
                client.model.to(args.device)
                q = Queue()
                client.local_train(q)
                with torch.no_grad():
                    client_weight = torch.cat(
                        [p.data.view(-1) for p in client.model.parameters()]
                    ).cpu()
                    delta_weight = initial_weight_flat - client_weight
                
                # Free up GPU memory immediately
                # Also, we no longer need the trained model state in the client object
                client.model.to('cpu')

                # Normalize the delta
                delta_np = delta_weight.numpy()
                row_norm = np.linalg.norm(delta_np)
                normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
                
                # Save the normalized delta to a temporary file
                delta_filepath = os.path.join(temp_dir, f"delta_{client_num - 1}.npy")
                np.save(delta_filepath, normalized_delta)

                # Add the delta to our in-memory batch for fitting
                delta_batch.append(normalized_delta)

                # When the batch is full, fit it and clear the in-memory batch
                if len(delta_batch) == BATCH_SIZE:
                    print(f"  > Fitting PCA with a batch of {len(delta_batch)} deltas...")
                    pca.partial_fit(np.array(delta_batch))
                    delta_batch.clear()

            # Process any remaining deltas from the last batch
            if len(delta_batch) >= N_COMPONENTS:
                print(f"  > Fitting PCA with the final batch of {len(delta_batch)} deltas...")
                pca.partial_fit(np.array(delta_batch))
                delta_batch.clear()

            print("PCA model has been fitted.")

            # --- PASS 2: Load Deltas from Disk and Transform ---
            print("Loading deltas from disk and transforming...")
            reduced_deltas = np.zeros((len(clientlist), N_COMPONENTS))
            for i in range(len(clientlist)):
                # Load the saved delta from disk
                delta_filepath = os.path.join(temp_dir, f"delta_{i}.npy")
                normalized_delta = np.load(delta_filepath).reshape(1, -1)
                
                # Transform the single delta and store it
                reduced_deltas[i] = pca.transform(normalized_delta)
            
            print("All deltas have been transformed.")

            # --- Step 3: Run KMeans ---
            print("Running KMeans on reduced deltas...")
            k = args.clusternum
            kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
            cluster_assignments = kmeans.labels_

            print("Cluster assignments:", cluster_assignments)

        finally:
            # --- CLEANUP ---
            # Always remove the temporary directory and its contents
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


        new_cluster_assignments = [-1] * args.clientnum
        roundrobincluster = 0

        for clusterind in range(args.clusternum):
            for clientind in range(args.clientnum):
                if cluster_assignments[clientind] == clusterind:
                    new_cluster_assignments[clientind] = roundrobincluster
                    roundrobincluster = (roundrobincluster + 1) % args.clusternum

        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            ind = 0
            for clientind in range(len(clientlist)):
                if new_cluster_assignments[clientind] == clusterind:
                    client = clientlist[clientind]
                    client.clusterid = clusterind
                    client.clientid = ind
                    ind +=1
                    cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)

        return centralserver
                                 
    elif args.clusteringtype == "clusterbygradientdissimilarityandsystemsimilarity":

        # Assuming 'centralserver', 'clientlist', and 'args' are defined

        print("Clustering by gradient dissimilarity...")

        # --- SETUP ---
        # Get the initial model state
        initial_weight_dict = centralserver.model.state_dict()
        with torch.no_grad():
            initial_weight_flat = torch.cat(
                [p.view(-1) for p in initial_weight_dict.values()]
            ).cpu()

        # Define PCA and batch size
        N_COMPONENTS = 20
        BATCH_SIZE = 20  # Must be >= n_components
        pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)

        # Create a temporary directory to store deltas
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for deltas: {temp_dir}")

        try:
            # --- PASS 1: Fit PCA and Save Deltas to Disk ---
            print("Fitting PCA model and caching deltas to disk...")
            delta_batch = []
            client_num = 0
            for client in clientlist:
                client_num += 1
                print(f"Processing Client {client_num}/{len(clientlist)}...")

                # Perform local training
                client.model.load_state_dict(initial_weight_dict)
                client.model.to(args.device)
                q = Queue()
                client.local_train(q)
                with torch.no_grad():
                    client_weight = torch.cat(
                        [p.data.view(-1) for p in client.model.parameters()]
                    ).cpu()
                    delta_weight = initial_weight_flat - client_weight
                
                # Free up GPU memory immediately
                # Also, we no longer need the trained model state in the client object
                client.model.to('cpu')

                # Normalize the delta
                delta_np = delta_weight.numpy()
                row_norm = np.linalg.norm(delta_np)
                normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
                
                # Save the normalized delta to a temporary file
                delta_filepath = os.path.join(temp_dir, f"delta_{client_num - 1}.npy")
                np.save(delta_filepath, normalized_delta)

                # Add the delta to our in-memory batch for fitting
                delta_batch.append(normalized_delta)

                # When the batch is full, fit it and clear the in-memory batch
                if len(delta_batch) == BATCH_SIZE:
                    print(f"  > Fitting PCA with a batch of {len(delta_batch)} deltas...")
                    pca.partial_fit(np.array(delta_batch))
                    delta_batch.clear()

            # Process any remaining deltas from the last batch
            if len(delta_batch) >= N_COMPONENTS:
                print(f"  > Fitting PCA with the final batch of {len(delta_batch)} deltas...")
                pca.partial_fit(np.array(delta_batch))
                delta_batch.clear()

            print("PCA model has been fitted.")

            # --- PASS 2: Load Deltas from Disk and Transform ---
            print("Loading deltas from disk and transforming...")
            reduced_deltas = np.zeros((len(clientlist), N_COMPONENTS))
            for i in range(len(clientlist)):
                # Load the saved delta from disk
                delta_filepath = os.path.join(temp_dir, f"delta_{i}.npy")
                normalized_delta = np.load(delta_filepath).reshape(1, -1)
                
                # Transform the single delta and store it
                reduced_deltas[i] = pca.transform(normalized_delta)
            
            print("All deltas have been transformed.")

            # --- Step 3: Run KMeans ---
            print("Running KMeans on reduced deltas...")
            k = args.clusternum
            kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
            cluster_assignments = kmeans.labels_

            print("Cluster assignments:", cluster_assignments)

        finally:
            # --- CLEANUP ---
            # Always remove the temporary directory and its contents
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


        new_cluster_assignments = [-1] * args.clientnum
        roundrobincluster = 0

        for clusterind in range(args.clusternum):
            indices = []
            for clientind in range(args.clientnum):
                if cluster_assignments[clientind] == clusterind:
                    indices.append(clientind)
            sortedindices = sorted(indices, key=lambda x: clientlist[x].calculate_training_time())
            for i in range(len(sortedindices)):
                new_cluster_assignments[sortedindices[i]] = i

        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            ind = 0
            for clientind in range(len(clientlist)):
                if new_cluster_assignments[clientind] == clusterind:
                    client = clientlist[clientind]
                    client.clusterid = clusterind
                    client.clientid = ind
                    ind +=1
                    cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)

        return centralserver
    
    elif args.clusteringtype == "clusterbysystemsimilarity":

        clienttrainingtimelist = []
        for client in clientlist:
            clienttrainingtimelist.append(client.calculate_training_time())
        clienttrainingtimearray = np.array(clienttrainingtimelist)
        clienttrainingtimearrayreshaped = clienttrainingtimearray.reshape(-1, 1)

        # Create KMeans instance with 2 clusters
        kmeans = KMeans(n_clusters=args.clusternum, random_state=args.randomseed)
        kmeans.fit(clienttrainingtimearrayreshaped)

        # Get cluster labels and centroids
        cluster_assignments = kmeans.labels_

        # Assign clusters
        for clusterind in range(args.clusternum):
            cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            ind = 0
            for clientind in range(len(clientlist)):
                if cluster_assignments[clientind] == clusterind:
                    client = clientlist[clientind]
                    client.clusterid = clusterind
                    client.clientid = ind
                    ind +=1
                    cluster.clientlist.append(client)
            centralserver.clusterlist.append(cluster)

        return centralserver
        
    elif args.clusteringtype == "custom":

        raise ValueError("customize it plz")
    
    else:

        raise ValueError("clustering type not supported")


        