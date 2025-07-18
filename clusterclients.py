from sklearn.cluster import KMeans
from queue import Queue
import numpy as np
import torch
from sklearn.decomposition import PCA
from cluster import Cluster
from centralserver import CentralServer
import random
import logging


def cluster_clients(clientlist, args):

    centralserver = CentralServer(args.interclusteringtype, args.centralserverepoch, args, [])

    if args.clusteringtype == "clusterbyclientorder":

        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        num_clients = len(clientlist)
        
        # clustering, same cluster size
        cluster_assignments = np.zeros(num_clients, dtype=int)
        clients_per_cluster = (num_clients + k - 1) // k
        for client_idx in range(num_clients):
            cluster_assignments[client_idx] = client_idx // clients_per_cluster

        # evaluate clustering by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)


    elif args.clusteringtype == "clusterbyrandomshuffle":
       
        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        num_clients = len(clientlist)
        
        # clustering, same cluster size
        cluster_assignments = []
        assignments_per_cluster = num_clients // k
        for cluster_id in range(k):
            cluster_assignments.extend([cluster_id] * assignments_per_cluster)
        remainder = num_clients % k
        for i in range(remainder):
            cluster_assignments.append(i)
        random.shuffle(cluster_assignments)
        cluster_assignments = np.array(cluster_assignments)
        
        # evaluate clustering by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)


    elif args.clusteringtype == "clusterbygradientbetweenassumeonelabeldominant100clients":
       
        # safety check for assumption
        if len(clientlist) != 100 or args.clusternum != 10:
            raise ValueError("This clustering type is hardcoded for 100 clients and 10 clusters.")

        # prepare clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum

        # clustering, same cluster size
        cluster_assignments_list = [9] * 5
        for i in range(9):
            cluster_assignments_list.extend([i] * 10)
        cluster_assignments_list.extend([9] * 5)
        cluster_assignments = np.array(cluster_assignments_list)
        
        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)


    elif args.clusteringtype == "clusterbygradientsimilarity":

        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        
        # clustering, kmeans on pca vectors
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        cluster_assignments = kmeans.labels_

        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)

    elif args.clusteringtype == "clusterbygradientdissimilarity":
        
        # perpare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
       
        # clustering, kmeans on pca vectors, then round-robin assigning different cluster to same groups in kmeans
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_assignments = kmeans.labels_
        final_assignments = np.full(len(clientlist), -1, dtype=int)
        round_robin_counter = 0
        for group_id in range(k):
            clients_in_group = np.where(initial_assignments == group_id)[0]
            for client_idx in clients_in_group:
                final_assignments[client_idx] = round_robin_counter
                round_robin_counter = (round_robin_counter + 1) % k
        cluster_assignments = final_assignments

        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)
    
    elif args.clusteringtype == "clusterbygradientdissimilaritygreedy":
        print("Running a TWO-STAGE HYBRID clustering (Marginal Gain Strategy)...")
        args.balance_lambda = 1.0

        # Standard prep step
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        
        k = args.clusternum
        N_COMPONENTS = reduced_deltas.shape[1]
        
        # --- STAGE 1: Initial Similarity Grouping (KMeans) ---
        print("STAGE 1: Running KMeans to find initial similarity groups...")
        M = k 
        kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_groups = kmeans.labels_ 
        group_centroids = kmeans.cluster_centers_

        # --- STAGE 2: Hybrid Greedy Assignment (Client by Client) ---
        print("STAGE 2: Running hybrid greedy assignment (one client at a time)...")
        
        # Organize clients by group and sort by distance to their centroid
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

        # Initialize state for final clusters
        global_avg_reduced_delta = np.average(reduced_deltas, axis=0, weights=client_datasizes)
        cluster_assignments = np.full(len(clientlist), -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        final_cluster_small_group_origins = [set() for _ in range(k)]

        # Sequentially process each client from the sorted groups
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

                # Normalize and combine costs
                q_min, q_max = raw_delta_quality_costs.min(), raw_delta_quality_costs.max()
                norm_quality_costs = (raw_delta_quality_costs - q_min) / (q_max - q_min + 1e-9)

                b_min, b_max = raw_balance_penalties.min(), raw_balance_penalties.max()
                norm_balance_penalties = np.zeros(k)
                if b_max - b_min > 1e-9:
                    norm_balance_penalties = (raw_balance_penalties - b_min) / (b_max - b_min)
                
                total_costs = norm_quality_costs + args.balance_lambda * norm_balance_penalties
                
                # Apply diversity penalty
                is_from_small_group = len(organized_clients[group_id]) <= k
                if is_from_small_group:
                    for j in range(k):
                        if len(final_cluster_small_group_origins[j]) > 0 and group_id not in final_cluster_small_group_origins[j]:
                            total_costs[j] += 1.0 
                
                best_cluster_idx = np.argmin(total_costs)
                cluster_assignments[client_idx] = best_cluster_idx
                
                # Update cluster stats
                cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                cluster_sum_datasizes[best_cluster_idx] += client_size
                cluster_client_counts[best_cluster_idx] += 1
                if is_from_small_group:
                    final_cluster_small_group_origins[best_cluster_idx].add(group_id)
        
        print("Hybrid assignment (marginal gain) complete.")
        print("Final cluster client counts:", cluster_client_counts)
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)
    


    # elif args.clusteringtype == "clusterbygradientdissimilaritygreedy":
    #     print("Running a TWO-STAGE HYBRID clustering (Marginal Gain Strategy)...")
    #     args.balance_lambda = 1.0
    #     reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        
    #     k = args.clusternum
    #     N_COMPONENTS = reduced_deltas.shape[1] # Use actual number of components from PCA
        
    #     print("STAGE 1: Running KMeans to find initial similarity groups...")
    #     M = k 
    #     kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
    #     initial_groups = kmeans.labels_ 
    #     group_centroids = kmeans.cluster_centers_

    #     print("STAGE 2: Running hybrid greedy assignment (marginal gain strategy)...")
    #     organized_clients = [[] for _ in range(M)]
    #     for client_idx, group_id in enumerate(initial_groups):
    #         organized_clients[group_id].append(client_idx)
    #     for group_id in range(M):
    #         clients_in_group = organized_clients[group_id]
    #         if not clients_in_group: continue
    #         centroid = group_centroids[group_id]
    #         distances = [np.linalg.norm(reduced_deltas[c_idx] - centroid) for c_idx in clients_in_group]
    #         sorted_clients = [c_idx for _, c_idx in sorted(zip(distances, clients_in_group))]
    #         organized_clients[group_id] = sorted_clients

    #     global_avg_reduced_delta = np.average(reduced_deltas, axis=0, weights=client_datasizes)
    #     cluster_assignments = np.full(len(clientlist), -1, dtype=int)
    #     cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS))
    #     cluster_sum_datasizes = np.zeros(k)
    #     cluster_client_counts = np.zeros(k, dtype=int)
    #     final_cluster_small_group_origins = [set() for _ in range(k)]

    #     for group_id in range(M):
    #         for client_idx in organized_clients[group_id]:
    #             client_delta = reduced_deltas[client_idx]
    #             client_size = client_datasizes[client_idx]
    #             raw_delta_quality_costs = np.zeros(k) 
    #             raw_balance_penalties = np.zeros(k)
    #             current_cluster_quality_costs = np.zeros(k)
    #             for j in range(k):
    #                 if cluster_client_counts[j] > 0:
    #                     current_avg = cluster_sum_weighted_deltas[j] / cluster_sum_datasizes[j]
    #                     current_cluster_quality_costs[j] = np.linalg.norm(current_avg - global_avg_reduced_delta)**2
    #             for j in range(k):
    #                 hypothetical_sum_weighted_delta = cluster_sum_weighted_deltas[j] + client_size * client_delta
    #                 hypothetical_sum_datasize = cluster_sum_datasizes[j] + client_size
    #                 hypothetical_avg = hypothetical_sum_weighted_delta / (hypothetical_sum_datasize + 1e-9)
    #                 new_quality_cost = np.linalg.norm(hypothetical_avg - global_avg_reduced_delta)**2
    #                 raw_delta_quality_costs[j] = new_quality_cost - current_cluster_quality_costs[j]
    #                 raw_balance_penalties[j] = cluster_client_counts[j]
    #             q_min, q_max = raw_delta_quality_costs.min(), raw_delta_quality_costs.max()
    #             norm_quality_costs = (raw_delta_quality_costs - q_min) / (q_max - q_min + 1e-9)
    #             b_min, b_max = raw_balance_penalties.min(), raw_balance_penalties.max()
    #             norm_balance_penalties = np.zeros(k)
    #             if b_max - b_min > 1e-9:
    #                 norm_balance_penalties = (raw_balance_penalties - b_min) / (b_max - b_min)
    #             total_costs = norm_quality_costs + args.balance_lambda * norm_balance_penalties
    #             is_from_small_group = len(organized_clients[group_id]) <= k
    #             if is_from_small_group:
    #                 for j in range(k):
    #                     if len(final_cluster_small_group_origins[j]) > 0 and group_id not in final_cluster_small_group_origins[j]:
    #                         total_costs[j] += 1.0 
    #             best_cluster_idx = np.argmin(total_costs)
    #             cluster_assignments[client_idx] = best_cluster_idx
    #             cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
    #             cluster_sum_datasizes[best_cluster_idx] += client_size
    #             cluster_client_counts[best_cluster_idx] += 1
    #             if is_from_small_group:
    #                 final_cluster_small_group_origins[best_cluster_idx].add(group_id)
        
    #     print("Hybrid assignment (marginal gain) complete.")
    #     print("Final cluster client counts:", cluster_client_counts)
    #     evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)

    elif args.clusteringtype == "clusterbysystemsimilarity":
        
        # perpare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        
        # clustering, kmeans based on training time
        client_training_times = []
        for client in clientlist:
            # Assuming calculate_training_time is a fast operation
            client_training_times.append(client.calculate_training_time())
        client_training_times_np = np.array(client_training_times).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(client_training_times_np)
        
        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)

    else:
        raise ValueError(f"Clustering type '{args.clusteringtype}' not supported")

    # actual assignment of clients and clusters
    centralserver.clusterlist = []
    for cluster_id in range(k):
        cluster = Cluster(cluster_id, args.clustercommunicationtime, args.intraclusteringtype, args.clusterepoch, args, [])
        centralserver.clusterlist.append(cluster)
    for client_idx, cluster_id in enumerate(cluster_assignments):
        client = clientlist[client_idx]
        client.clusterid = int(cluster_id)
        centralserver.clusterlist[int(cluster_id)].clientlist.append(client)  
    for cluster in centralserver.clusterlist:
        for intra_cluster_id, client in enumerate(cluster.clientlist):
            client.clientid = intra_cluster_id
            
    return centralserver



def prepare_and_run_pca(clientlist, centralserver, args):
    """
    Performs local training for all clients to get model deltas,
    then runs PCA.
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
    N_COMPONENTS = 100
    pca = PCA(n_components=N_COMPONENTS)
    reduced_deltas = pca.fit_transform(all_deltas_np)
    print(f"PCA complete, using {N_COMPONENTS} components.")
    
    return reduced_deltas, client_datasizes_np, all_deltas_np

def evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes):
    """
    Calculates and prints inter-cluster and intra-cluster variance.
    """
    print("\n--- Clustering Evaluation Metrics (on Original Gradients) ---")
    
    # --- Inter-Cluster Variance (Weighted) ---
    print("1. Inter-Cluster Variance (Weighted):")
    cluster_averages = []
    cluster_total_datasizes = []
    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 0:
            deltas_in_cluster = original_deltas[indices_in_cluster]
            datasizes_in_cluster = client_datasizes[indices_in_cluster]
            weighted_avg_grad = np.average(deltas_in_cluster, axis=0, weights=datasizes_in_cluster)
            cluster_averages.append(weighted_avg_grad)
            cluster_total_datasizes.append(np.sum(datasizes_in_cluster))

    if len(cluster_averages) < 2:
        logging.info("  - N/A (requires at least 2 non-empty clusters)")
    else:
        cluster_averages_np = np.array(cluster_averages)
        global_average_gradient = np.average(cluster_averages_np, axis=0, weights=cluster_total_datasizes)
        squared_distances = np.sum((cluster_averages_np - global_average_gradient)**2, axis=1)
        weighted_inter_variance = np.average(squared_distances, weights=cluster_total_datasizes)
        logging.info(f"  - Weighted Variance of Gradients: {weighted_inter_variance:.6f} (lower is better)")

    # --- Intra-Cluster Variance ---
    print("2. Intra-Cluster Variance:")
    intra_cluster_variances = []
    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 1:
            deltas_in_cluster = original_deltas[indices_in_cluster]
            mean_of_deltas = np.mean(deltas_in_cluster, axis=0)
            squared_distances = np.sum((deltas_in_cluster - mean_of_deltas)**2, axis=1)
            intra_cluster_variances.append(np.mean(squared_distances))
        else:
            intra_cluster_variances.append(0.0)
            
    mean_intra_variance = np.mean(intra_cluster_variances)
    logging.info(f"  - Per-Cluster Variances: {[float(f'{v:.6f}') for v in intra_cluster_variances]}")
    logging.info(f"  - Mean Intra-Cluster Variance: {mean_intra_variance:.6f} (lower is better)")