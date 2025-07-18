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
    elif args.clusteringtype == "clusterbygradientbetweenassumeonelabeldominant100clients":

        cluster_assignments = [9] * 5

        for i in range(9):
            for _ in range(10):
                cluster_assignments.append(i)
        cluster_assignments.extend([9,9,9,9,9])

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

        # for clusterind in range(args.clusternum):
        

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
    




    elif args.clusteringtype == "clusterbygradientdissimilaritygreedy":
        print("Running a TWO-STAGE HYBRID clustering algorithm...")
        args.balance_lambda = 1.0

        # --- SETUP ---
        initial_weight_dict = centralserver.model.state_dict()
        with torch.no_grad():
            initial_weight_flat = torch.cat(
                [p.view(-1) for p in initial_weight_dict.values()]
            ).cpu()

        N_COMPONENTS = 20
        BATCH_SIZE = 20
        if len(clientlist) < BATCH_SIZE:
            BATCH_SIZE = max(len(clientlist), N_COMPONENTS)
            print(f"Warning: Number of clients is less than BATCH_SIZE. Adjusting BATCH_SIZE to {BATCH_SIZE}.")
        
        pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory for deltas: {temp_dir}")

        try:
            # --- PASS 1 & 2: Data Prep & PCA (No Change) ---
            print("PASS 1: Fitting PCA, caching deltas, and collecting data sizes...")
            delta_batch = []
            client_datasizes = []
            for client_num, client in enumerate(clientlist, 1):
                print(f"\rProcessing Client {client_num}/{len(clientlist)}...", end='')
                datasize = len(client.dataloader)
                client_datasizes.append(datasize)
                client.model.load_state_dict(initial_weight_dict)
                client.model.to(args.device)
                q = Queue()
                client.local_train(q)
                with torch.no_grad():
                    client_weight = torch.cat([p.data.view(-1) for p in client.model.parameters()]).cpu()
                    delta_weight = initial_weight_flat - client_weight
                client.model.to('cpu')
                delta_np = delta_weight.numpy()
                row_norm = np.linalg.norm(delta_np)
                normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
                delta_filepath = os.path.join(temp_dir, f"delta_{client_num - 1}.npy")
                np.save(delta_filepath, normalized_delta)
                delta_batch.append(normalized_delta)
                if len(delta_batch) == BATCH_SIZE:
                    pca.partial_fit(np.array(delta_batch))
                    delta_batch.clear()
            print()
            if len(delta_batch) >= N_COMPONENTS:
                pca.partial_fit(np.array(delta_batch))
            client_datasizes = np.array(client_datasizes)
            print("PCA model fitted and data sizes collected.")

            print("PASS 2: Loading deltas from disk and transforming...")
            reduced_deltas = np.zeros((len(clientlist), N_COMPONENTS))
            for i in range(len(clientlist)):
                delta_filepath = os.path.join(temp_dir, f"delta_{i}.npy")
                normalized_delta = np.load(delta_filepath).reshape(1, -1)
                reduced_deltas[i] = pca.transform(normalized_delta)
            print("All deltas have been transformed.")

            # --- STAGE 1: Initial Similarity Grouping (KMeans) ---
            print("STAGE 1: Running KMeans to find initial similarity groups...")
            # Number of initial groups. Can be same as final clusters or larger.
            M = args.clusternum 
            kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
            initial_groups = kmeans.labels_

            # --- STAGE 2: Hybrid Greedy Re-assignment ---
            print("STAGE 2: Running hybrid greedy assignment...")
            k = args.clusternum
            
            # Organize clients by their initial KMeans group
            organized_clients = [[] for _ in range(M)]
            for client_idx, group_id in enumerate(initial_groups):
                organized_clients[group_id].append(client_idx)

            # Initialize state for final clusters
            global_avg_reduced_delta = np.average(reduced_deltas, axis=0, weights=client_datasizes)
            cluster_assignments = np.full(len(clientlist), -1, dtype=int)
            cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS))
            cluster_sum_datasizes = np.zeros(k)
            cluster_client_counts = np.zeros(k, dtype=int)

            # Iterate round-robin through the initial groups
            max_group_size = max(len(g) for g in organized_clients) if organized_clients else 0
            for i in range(max_group_size):
                for group_id in range(M):
                    if i < len(organized_clients[group_id]):
                        client_idx = organized_clients[group_id][i]
                        
                        # --- Normalized & Balanced Cost Calculation ---
                        client_delta = reduced_deltas[client_idx]
                        client_size = client_datasizes[client_idx]
                        raw_quality_costs = np.zeros(k)
                        raw_balance_penalties = np.zeros(k)

                        for j in range(k):
                            hypothetical_sum_weighted_delta = cluster_sum_weighted_deltas[j] + client_size * client_delta
                            hypothetical_sum_datasize = cluster_sum_datasizes[j] + client_size
                            hypothetical_avg = hypothetical_sum_weighted_delta / (hypothetical_sum_datasize + 1e-9)
                            raw_quality_costs[j] = np.linalg.norm(hypothetical_avg - global_avg_reduced_delta)**2
                            raw_balance_penalties[j] = cluster_client_counts[j]

                        q_min, q_max = raw_quality_costs.min(), raw_quality_costs.max()
                        norm_quality_costs = (raw_quality_costs - q_min) / (q_max - q_min + 1e-9)

                        b_min, b_max = raw_balance_penalties.min(), raw_balance_penalties.max()
                        norm_balance_penalties = np.zeros(k)
                        if b_max - b_min > 1e-9:
                            norm_balance_penalties = (raw_balance_penalties - b_min) / (b_max - b_min)
                        
                        total_costs = norm_quality_costs + args.balance_lambda * norm_balance_penalties
                        best_cluster_idx = np.argmin(total_costs)
                        # --- End Cost Calculation ---

                        # Assign client and update stats
                        cluster_assignments[client_idx] = best_cluster_idx
                        cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                        cluster_sum_datasizes[best_cluster_idx] += client_size
                        cluster_client_counts[best_cluster_idx] += 1
            
            print("Hybrid assignment complete.")
            print("Final cluster client counts:", cluster_client_counts)

            # --- Final Step: Create Cluster Objects ---
            print("Finalizing server's cluster list...")
            for cluster_id in range(k):
                cluster = Cluster(cluster_id, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
                centralserver.clusterlist.append(cluster)
            
            for client_idx, cluster_id in enumerate(cluster_assignments):
                client = clientlist[client_idx]
                client.clusterid = cluster_id
                centralserver.clusterlist[cluster_id].clientlist.append(client)
                
            for cluster in centralserver.clusterlist:
                for intra_cluster_id, client in enumerate(cluster.clientlist):
                    client.clientid = intra_cluster_id

        finally:
            # --- CLEANUP ---
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        return centralserver
    





                                 
    elif args.clusteringtype == "clusterbygradientdissimilarityandsystemsimilarity":
        
        raise ValueError("to be implemented")

        # # Assuming 'centralserver', 'clientlist', and 'args' are defined

        # print("Clustering by gradient dissimilarity...")

        # # --- SETUP ---
        # # Get the initial model state
        # initial_weight_dict = centralserver.model.state_dict()
        # with torch.no_grad():
        #     initial_weight_flat = torch.cat(
        #         [p.view(-1) for p in initial_weight_dict.values()]
        #     ).cpu()

        # # Define PCA and batch size
        # N_COMPONENTS = 20
        # BATCH_SIZE = 20  # Must be >= n_components
        # pca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=BATCH_SIZE)

        # # Create a temporary directory to store deltas
        # temp_dir = tempfile.mkdtemp()
        # print(f"Using temporary directory for deltas: {temp_dir}")

        # try:
        #     # --- PASS 1: Fit PCA and Save Deltas to Disk ---
        #     print("Fitting PCA model and caching deltas to disk...")
        #     delta_batch = []
        #     client_num = 0
        #     for client in clientlist:
        #         client_num += 1
        #         print(f"Processing Client {client_num}/{len(clientlist)}...")

        #         # Perform local training
        #         client.model.load_state_dict(initial_weight_dict)
        #         client.model.to(args.device)
        #         q = Queue()
        #         client.local_train(q)
        #         with torch.no_grad():
        #             client_weight = torch.cat(
        #                 [p.data.view(-1) for p in client.model.parameters()]
        #             ).cpu()
        #             delta_weight = initial_weight_flat - client_weight
                
        #         # Free up GPU memory immediately
        #         # Also, we no longer need the trained model state in the client object
        #         client.model.to('cpu')

        #         # Normalize the delta
        #         delta_np = delta_weight.numpy()
        #         row_norm = np.linalg.norm(delta_np)
        #         normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
                
        #         # Save the normalized delta to a temporary file
        #         delta_filepath = os.path.join(temp_dir, f"delta_{client_num - 1}.npy")
        #         np.save(delta_filepath, normalized_delta)

        #         # Add the delta to our in-memory batch for fitting
        #         delta_batch.append(normalized_delta)

        #         # When the batch is full, fit it and clear the in-memory batch
        #         if len(delta_batch) == BATCH_SIZE:
        #             print(f"  > Fitting PCA with a batch of {len(delta_batch)} deltas...")
        #             pca.partial_fit(np.array(delta_batch))
        #             delta_batch.clear()

        #     # Process any remaining deltas from the last batch
        #     if len(delta_batch) >= N_COMPONENTS:
        #         print(f"  > Fitting PCA with the final batch of {len(delta_batch)} deltas...")
        #         pca.partial_fit(np.array(delta_batch))
        #         delta_batch.clear()

        #     print("PCA model has been fitted.")

        #     # --- PASS 2: Load Deltas from Disk and Transform ---
        #     print("Loading deltas from disk and transforming...")
        #     reduced_deltas = np.zeros((len(clientlist), N_COMPONENTS))
        #     for i in range(len(clientlist)):
        #         # Load the saved delta from disk
        #         delta_filepath = os.path.join(temp_dir, f"delta_{i}.npy")
        #         normalized_delta = np.load(delta_filepath).reshape(1, -1)
                
        #         # Transform the single delta and store it
        #         reduced_deltas[i] = pca.transform(normalized_delta)
            
        #     print("All deltas have been transformed.")

        #     # --- Step 3: Run KMeans ---
        #     print("Running KMeans on reduced deltas...")
        #     k = args.clusternum
        #     kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        #     cluster_assignments = kmeans.labels_

        #     print("Cluster assignments:", cluster_assignments)

        # finally:
        #     # --- CLEANUP ---
        #     # Always remove the temporary directory and its contents
        #     print(f"Cleaning up temporary directory: {temp_dir}")
        #     shutil.rmtree(temp_dir)


        # new_cluster_assignments = [-1] * args.clientnum
        # roundrobincluster = 0

        # for clusterind in range(args.clusternum):
        #     indices = []
        #     for clientind in range(args.clientnum):
        #         if cluster_assignments[clientind] == clusterind:
        #             indices.append(clientind)
        #     sortedindices = sorted(indices, key=lambda x: clientlist[x].calculate_training_time())
        #     for i in range(len(sortedindices)):
        #         new_cluster_assignments[sortedindices[i]] = i

        # for clusterind in range(args.clusternum):
        #     cluster = Cluster(clusterind, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
        #     ind = 0
        #     for clientind in range(len(clientlist)):
        #         if new_cluster_assignments[clientind] == clusterind:
        #             client = clientlist[clientind]
        #             client.clusterid = clusterind
        #             client.clientid = ind
        #             ind +=1
        #             cluster.clientlist.append(client)
        #     centralserver.clusterlist.append(cluster)

        return centralserver
    
    elif args.clusteringtype == "clusterbysystemsimilarity":

        clienttrainingtimelist = []
        for client in clientlist:
            clienttrainingtimelist.append(client.calculate_training_time())
        clienttrainingtimearray = np.array(clienttrainingtimelist)
        clienttrainingtimearrayreshaped = clienttrainingtimearray.reshape(-1, 1)

        # Create KMeans instance with 2 clusters
        kmeans = KMeans(n_clusters=args.clusternum, random_state=args.randomseed, n_init="auto")
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


        