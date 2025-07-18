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
import logging

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
    
    if args.clusteringtype == "clusterbygradientsimilarity":

        print("Clustering by 'clusterbygradientsimilarity' using PCA and KMeans...")
        reduced_deltas, client_datasizes = prepare_and_run_pca(clientlist, centralserver, args)
        
        k = args.clusternum
        print("Running KMeans on reduced deltas...")
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        cluster_assignments = kmeans.labels_
        print("Cluster assignments:", cluster_assignments)

        evaluate_clustering(k, cluster_assignments, reduced_deltas, client_datasizes)

        print("\nFinalizing server's cluster list...")
        centralserver.clusterlist = []
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
        return centralserver

    elif args.clusteringtype == "clusterbygradientdissimilarity":

        print("Clustering by 'clusterbygradientdissimilarity' using KMeans then Round-Robin...")
        reduced_deltas, client_datasizes = prepare_and_run_pca(clientlist, centralserver, args)

        k = args.clusternum
        print("Running KMeans to find initial similarity groups...")
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_assignments = kmeans.labels_
        
        # --- Round-Robin Re-assignment to enforce diversity ---
        print("Re-assigning clients via round-robin...")
        final_assignments = np.full(len(clientlist), -1, dtype=int)
        round_robin_counter = 0
        # Process clients group by group from KMeans result
        for group_id in range(k):
            clients_in_group = np.where(initial_assignments == group_id)[0]
            for client_idx in clients_in_group:
                final_assignments[client_idx] = round_robin_counter
                round_robin_counter = (round_robin_counter + 1) % k
        print("Final cluster assignments:", final_assignments)

        # Evaluate the final assignments after the round-robin step
        evaluate_clustering(k, final_assignments, reduced_deltas, client_datasizes)

        print("\nFinalizing server's cluster list...")
        centralserver.clusterlist = []
        for cluster_id in range(k):
            cluster = Cluster(cluster_id, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
            centralserver.clusterlist.append(cluster)
        for client_idx, cluster_id in enumerate(final_assignments):
            client = clientlist[client_idx]
            client.clusterid = cluster_id
            centralserver.clusterlist[cluster_id].clientlist.append(client)
        for cluster in centralserver.clusterlist:
            for intra_cluster_id, client in enumerate(cluster.clientlist):
                client.clientid = intra_cluster_id
        return centralserver

    elif args.clusteringtype == "clusterbygradientdissimilaritygreedy":

        print("Running a TWO-STAGE HYBRID clustering (Marginal Gain Strategy)...")
        args.balance_lambda = 1.0

        reduced_deltas, client_datasizes = prepare_and_run_pca(clientlist, centralserver, args)
        
        N_COMPONENTS = 20
        k = args.clusternum
        
        print("STAGE 1: Running KMeans to find initial similarity groups...")
        M = k 
        kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_groups = kmeans.labels_ 
        group_centroids = kmeans.cluster_centers_

        print("STAGE 2: Running hybrid greedy assignment (marginal gain strategy)...")
        organized_clients = [[] for _ in range(M)]
        for client_idx, group_id in enumerate(initial_groups):
            organized_clients[group_id].append(client_idx)
        for group_id in range(M):
            clients_in_group = organized_clients[group_id]
            if not clients_in_group: continue
            centroid = group_centroids[group_id]
            distances = [np.linalg.norm(reduced_deltas[c_idx] - centroid) for c_idx in clients_in_group]
            sorted_clients = [c_idx for _, c_idx in sorted(zip(distances, clients_in_group))]
            organized_clients[group_id] = sorted_clients

        global_avg_reduced_delta = np.average(reduced_deltas, axis=0, weights=client_datasizes)
        cluster_assignments = np.full(len(clientlist), -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        final_cluster_small_group_origins = [set() for _ in range(k)]

        for group_id in range(M):
            for client_idx in organized_clients[group_id]:
                client_delta = reduced_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                raw_delta_quality_costs = np.zeros(k) 
                raw_balance_penalties = np.zeros(k)
                current_cluster_quality_costs = np.zeros(k)
                for j in range(k):
                    if cluster_client_counts[j] > 0:
                        current_avg = cluster_sum_weighted_deltas[j] / cluster_sum_datasizes[j]
                        current_cluster_quality_costs[j] = np.linalg.norm(current_avg - global_avg_reduced_delta)**2
                for j in range(k):
                    hypothetical_sum_weighted_delta = cluster_sum_weighted_deltas[j] + client_size * client_delta
                    hypothetical_sum_datasize = cluster_sum_datasizes[j] + client_size
                    hypothetical_avg = hypothetical_sum_weighted_delta / (hypothetical_sum_datasize + 1e-9)
                    new_quality_cost = np.linalg.norm(hypothetical_avg - global_avg_reduced_delta)**2
                    raw_delta_quality_costs[j] = new_quality_cost - current_cluster_quality_costs[j]
                    raw_balance_penalties[j] = cluster_client_counts[j]
                q_min, q_max = raw_delta_quality_costs.min(), raw_delta_quality_costs.max()
                norm_quality_costs = (raw_delta_quality_costs - q_min) / (q_max - q_min + 1e-9)
                b_min, b_max = raw_balance_penalties.min(), raw_balance_penalties.max()
                norm_balance_penalties = np.zeros(k)
                if b_max - b_min > 1e-9:
                    norm_balance_penalties = (raw_balance_penalties - b_min) / (b_max - b_min)
                total_costs = norm_quality_costs + args.balance_lambda * norm_balance_penalties
                is_from_small_group = len(organized_clients[group_id]) <= k
                if is_from_small_group:
                    for j in range(k):
                        if len(final_cluster_small_group_origins[j]) > 0 and group_id not in final_cluster_small_group_origins[j]:
                            total_costs[j] += 1.0 
                best_cluster_idx = np.argmin(total_costs)
                cluster_assignments[client_idx] = best_cluster_idx
                cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                cluster_sum_datasizes[best_cluster_idx] += client_size
                cluster_client_counts[best_cluster_idx] += 1
                if is_from_small_group:
                    final_cluster_small_group_origins[best_cluster_idx].add(group_id)
        
        print("Hybrid assignment (marginal gain) complete.")
        print("Final cluster client counts:", cluster_client_counts)
        
        evaluate_clustering(k, cluster_assignments, reduced_deltas, client_datasizes)

        print("\nFinalizing server's cluster list...")
        centralserver.clusterlist = []
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
        return centralserver



                                 
    elif args.clusteringtype == "clusterbygradientdissimilarityandsystemsimilarity":
        
        raise ValueError("to be implemented")

    
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










def prepare_and_run_pca(clientlist, centralserver, args):
    """
    Performs local training for all clients to get model deltas,
    then runs PCA to reduce their dimensionality.
    """
    print("Preparing deltas and running PCA...")
    all_deltas = []
    client_datasizes = []
    
    initial_weight_dict = centralserver.model.state_dict()
    with torch.no_grad():
        initial_weight_flat = torch.cat(
            [p.view(-1) for p in initial_weight_dict.values()]
        ).cpu()

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
        all_deltas.append(normalized_delta)
    print() 

    all_deltas_np = np.array(all_deltas)
    client_datasizes_np = np.array(client_datasizes)
    N_COMPONENTS = 20
    pca = PCA(n_components=N_COMPONENTS)
    reduced_deltas = pca.fit_transform(all_deltas_np)
    print("PCA complete.")
    return reduced_deltas, client_datasizes_np

def evaluate_clustering(k, cluster_assignments, reduced_deltas, client_datasizes):
    """
    Calculates and prints inter-cluster and intra-cluster variance.
    """
    # --- INTER-CLUSTER VARIANCE (Variance between clusters) ---
    print("\n--- Clustering Evaluation Metrics ---")
    print("1. Inter-Cluster Variance (how different clusters are from each other):")
    cluster_averages = []
    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 0:
            deltas_in_cluster = reduced_deltas[indices_in_cluster]
            datasizes_in_cluster = client_datasizes[indices_in_cluster]
            weighted_avg_grad = np.average(deltas_in_cluster, axis=0, weights=datasizes_in_cluster)
            cluster_averages.append(weighted_avg_grad)

    if len(cluster_averages) < 2:
        print("  - N/A (requires at least 2 non-empty clusters)")
    else:
        cluster_averages_np = np.array(cluster_averages)
        mean_of_cluster_averages = np.mean(cluster_averages_np, axis=0)
        squared_distances = np.sum((cluster_averages_np - mean_of_cluster_averages)**2, axis=1)
        inter_cluster_variance = np.mean(squared_distances)
        logging.info(f"  - Variance of Aggregated Gradients: {inter_cluster_variance:.6f} (lower is better)")

    # --- INTRA-CLUSTER VARIANCE (Variance within clusters) ---
    print("2. Intra-Cluster Variance (how similar clients are within each cluster):")
    intra_cluster_variances = []
    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 1:
            deltas_in_cluster = reduced_deltas[indices_in_cluster]
            mean_of_deltas = np.mean(deltas_in_cluster, axis=0)
            squared_distances = np.sum((deltas_in_cluster - mean_of_deltas)**2, axis=1)
            intra_cluster_variances.append(np.mean(squared_distances))
        else:
            intra_cluster_variances.append(0.0) # Variance is 0 for clusters with 0 or 1 client
            
    mean_intra_variance = np.mean(intra_cluster_variances)
    logging.info(f"  - Per-Cluster Variances: {[float(f'{v:.6f}') for v in intra_cluster_variances]}")
    logging.info(f"  - Mean Intra-Cluster Variance: {mean_intra_variance:.6f} (lower is better)")

        