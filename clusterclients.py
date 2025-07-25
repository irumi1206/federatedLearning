from sklearn.cluster import KMeans
from queue import Queue
import numpy as np
import torch
from sklearn.decomposition import PCA
from cluster import Cluster
from centralserver import CentralServer
import random
import logging
from collections import defaultdict


def cluster_clients(centralserver, clientlist, args):

    print(f"DEBUG: Received clustering type: '{args.clusteringtype}'")

    # centralserver = CentralServer(args.interclusteringtype, args.centralserverepoch, args, [])

    if args.clusteringtype == "clusterbyclientorder":

        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        num_clients = len(clientlist)
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        
        # clustering, same cluster size
        cluster_assignments = np.zeros(num_clients, dtype=int)
        clients_per_cluster = (num_clients + k - 1) // k
        for client_idx in range(num_clients):
            cluster_assignments[client_idx] = client_idx // clients_per_cluster

        # evaluate clustering by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "clusterbyrandomshuffle":
       
        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        num_clients = len(clientlist)
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        
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
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "clusterbygradientbetweenassumeonelabeldominant100clients":
       
        # safety check for assumption
        if len(clientlist) != 100 or args.clusternum != 10:
            raise ValueError("This clustering type is hardcoded for 100 clients and 10 clusters.")

        # prepare clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])

        # clustering, same cluster size
        cluster_assignments_list = [9] * 5
        for i in range(9):
            cluster_assignments_list.extend([i] * 10)
        cluster_assignments_list.extend([9] * 5)
        cluster_assignments = np.array(cluster_assignments_list)
        
        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "clusterbygradientsimilarity":

        # prepare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        
        # clustering, kmeans on pca vectors
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        cluster_assignments = kmeans.labels_

        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)

    elif args.clusteringtype == "clusterbygradientbetween":
        
        # perpare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        level = args.roundrobinlevel
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
       
        # clustering, kmeans on pca vectors, then round-robin assigning different cluster to same groups in kmeans
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_assignments = kmeans.labels_
        final_assignments = np.full(len(clientlist), -1, dtype=int)
        round_robin_counter = 0
        clusterassignflag = level
        for group_id in range(k):
            clients_in_group = np.where(initial_assignments == group_id)[0]
            for client_idx in clients_in_group:
                final_assignments[client_idx] = round_robin_counter
                clusterassignflag = clusterassignflag -1
                if clusterassignflag ==0:
                    clusterassignflag = level
                    round_robin_counter = (round_robin_counter + 1) % k

        cluster_assignments = final_assignments

        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)

    elif args.clusteringtype == "clusterbygradientdissimilarity":
        
        # perpare for clustering
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
       
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
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)






    # elif args.clusteringtype == "clusterbygradientdissimilaritygreedy":
    #     print("Running a FIVE-ASPECT HYBRID clustering...")

    #     # Standard prep step
    #     reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        
    #     k = args.clusternum
    #     N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
    #     num_clients = len(clientlist)
        
    #     # --- STAGE 1: Initial Similarity Grouping (KMeans on PCA space) ---
    #     print("STAGE 1: Running KMeans on reduced deltas to determine processing ORDER...")
    #     M = k 
    #     kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
    #     initial_groups = kmeans.labels_ 
    #     group_centroids = kmeans.cluster_centers_

    #     # --- STAGE 2: Hybrid Greedy Assignment ---
    #     print("STAGE 2: Running hybrid greedy assignment...")
        
    #     # Organize clients by group and sort by distance to their centroid
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

    #     # Initialize state for final clusters
    #     cluster_assignments = np.full(num_clients, -1, dtype=int)
    #     cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
    #     cluster_sum_datasizes = np.zeros(k)
    #     cluster_client_counts = np.zeros(k, dtype=int)
    #     final_cluster_group_counts = [defaultdict(int) for _ in range(k)]

    #     # 5. Process clients sequentially by KMeans group
    #     for group_id in range(M):
    #         for client_idx in organized_clients[group_id]:
    #             print(f"\rAssigning client {client_idx+1}/{num_clients}...", end='')
                
    #             client_delta = original_deltas[client_idx]
    #             client_size = client_datasizes[client_idx]
                
    #             raw_quality_scores = np.zeros(k)
    #             raw_balance_scores = np.zeros(k)
    #             raw_diversity_scores = np.zeros(k)

    #             for j in range(k):
    #                 hypothetical_counts = cluster_client_counts.copy()
    #                 hypothetical_counts[j] += 1
                    
    #                 # 1. Calculate the Quality Score (Variance of ORIGINAL Gradients)
    #                 hypothetical_cluster_avgs = []
    #                 for cluster_id in range(k):
    #                     if hypothetical_counts[cluster_id] > 0:
    #                         current_sum_deltas = cluster_sum_weighted_deltas[cluster_id]
    #                         current_sum_sizes = cluster_sum_datasizes[cluster_id]
    #                         if cluster_id == j:
    #                             new_sum_deltas = current_sum_deltas + (client_size * client_delta)
    #                             new_sum_sizes = current_sum_sizes + client_size
    #                             hypothetical_cluster_avgs.append(new_sum_deltas / new_sum_sizes)
    #                         else:
    #                             hypothetical_cluster_avgs.append(current_sum_deltas / current_sum_sizes)
                    
    #                 raw_quality_scores[j] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0

    #                 # 2. Calculate the Balance Score (Variance of Sizes)
    #                 raw_balance_scores[j] = np.var(hypothetical_counts)
                    
    #                 # 3. Calculate the Diversity Score (Linear penalty for same-group clients)
    #                 raw_diversity_scores[j] = final_cluster_group_counts[j][group_id]
                
    #             # 4. Normalize all three scores using Min-Max Scaling
    #             q_min, q_max = raw_quality_scores.min(), raw_quality_scores.max()
    #             norm_quality_scores = (raw_quality_scores - q_min) / (q_max - q_min + 1e-9)

    #             b_min, b_max = raw_balance_scores.min(), raw_balance_scores.max()
    #             norm_balance_scores = (raw_balance_scores - b_min) / (b_max - b_min + 1e-9)
                
    #             d_min, d_max = raw_diversity_scores.min(), raw_diversity_scores.max()
    #             norm_diversity_scores = (raw_diversity_scores - d_min) / (d_max - d_min + 1e-9) if d_max > d_min else np.zeros(k)
                
    #             total_costs = (norm_quality_scores + 
    #                         args.balancelambda * norm_balance_scores + args.diversitylambda * norm_diversity_scores)
                
    #             # --- ADDED: Print statements for debugging ---
    #             print(f"\n--- Assigning Client {client_idx} (from KMeans group {group_id}) ---")
    #             print(f"Raw Quality Scores : {np.round(raw_quality_scores, 4)}")
    #             print(f"Raw Balance Scores : {np.round(raw_balance_scores, 4)}")
    #             print(f"Raw Diversity Scores: {np.round(raw_diversity_scores, 4)}")
    #             print(f"Norm Quality Scores: {np.round(norm_quality_scores, 4)}")
    #             print(f"Norm Balance Scores: {np.round(norm_balance_scores, 4)}")
    #             print(f"Norm Diversity Scores: {np.round(norm_diversity_scores, 4)}")
    #             print(f"Total Costs        : {np.round(total_costs, 4)}")
    #             # --- End of added prints ---


    #             best_cluster_idx = np.argmin(total_costs)
    #             cluster_assignments[client_idx] = best_cluster_idx
                
    #             # Permanently update the state
    #             cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
    #             cluster_sum_datasizes[best_cluster_idx] += client_size
    #             cluster_client_counts[best_cluster_idx] += 1
    #             final_cluster_group_counts[best_cluster_idx][group_id] += 1
                    
    #     print("\nHybrid assignment complete.")
    #     print("Final cluster client counts:", cluster_client_counts)
        
    #     evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes)

    #     # ... (code to create final cluster objects) ...


    elif args.clusteringtype == "greedy_variance_minmax":

        k = args.clusternum
        num_clients = len(clientlist)
        cluster_assignments = None
        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])

        print("Running Greedy: Quality=Global Variance, Normalization=Min-Max")
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        
        M = k 
        kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_groups = kmeans.labels_ 
        group_centroids = kmeans.cluster_centers_

        organized_clients = [[] for _ in range(M)]
        for client_idx, group_id in enumerate(initial_groups): organized_clients[group_id].append(client_idx)
        for group_id in range(M):
            clients_in_group = organized_clients[group_id]
            if not clients_in_group: continue
            centroid = group_centroids[group_id]
            distances = [np.linalg.norm(reduced_deltas[c_idx] - centroid) for c_idx in clients_in_group]
            sorted_clients = [c_idx for _, c_idx in sorted(zip(distances, clients_in_group))]
            organized_clients[group_id] = sorted_clients

        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        final_cluster_group_counts = [defaultdict(int) for _ in range(k)]

        for group_id in range(M):
            for client_idx in organized_clients[group_id]:
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                raw_quality_scores, raw_balance_scores, raw_diversity_scores = np.zeros(k), np.zeros(k), np.zeros(k)
                for j in range(k):
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                    hypothetical_cluster_avgs = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            if cid == j:
                                avg = (sum_d + (client_size * client_delta)) / (sum_s + client_size)
                            else:
                                avg = sum_d / sum_s
                            hypothetical_cluster_avgs.append(avg)
                    raw_quality_scores[j] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0
                    raw_balance_scores[j] = np.var(hypothetical_counts)
                    raw_diversity_scores[j] = final_cluster_group_counts[j][group_id]
                
                q_min, q_max = raw_quality_scores.min(), raw_quality_scores.max()
                norm_q = (raw_quality_scores - q_min) / (q_max - q_min + 1e-9)
                b_min, b_max = raw_balance_scores.min(), raw_balance_scores.max()
                norm_b = (raw_balance_scores - b_min) / (b_max - b_min + 1e-9)
                d_min, d_max = raw_diversity_scores.min(), raw_diversity_scores.max()
                norm_d = (raw_diversity_scores - d_min) / (d_max - d_min + 1e-9) if d_max > d_min else np.zeros(k)
                
                total_costs = norm_q + args.balancelambda * norm_b + args.diversitylambda * norm_d
                best_cluster_idx = np.argmin(total_costs)
                cluster_assignments[client_idx] = best_cluster_idx
                cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                cluster_sum_datasizes[best_cluster_idx] += client_size
                cluster_client_counts[best_cluster_idx] += 1
                final_cluster_group_counts[best_cluster_idx][group_id] += 1
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    
    elif args.clusteringtype == "greedy_margin_minmax":

        k = args.clusternum
        num_clients = len(clientlist)
        cluster_assignments = None

        # Pre-calculate all client training times
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        
        print("Running Greedy: Quality=Marginal Gain, Normalization=Min-Max")
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]

        M = k 
        kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_groups, group_centroids = kmeans.labels_, kmeans.cluster_centers_
        organized_clients = [[] for _ in range(M)]
        for idx, gid in enumerate(initial_groups): organized_clients[gid].append(idx)
        for gid in range(M):
            if not organized_clients[gid]: continue
            dists = [np.linalg.norm(reduced_deltas[cid] - group_centroids[gid]) for cid in organized_clients[gid]]
            organized_clients[gid] = [cid for _, cid in sorted(zip(dists, organized_clients[gid]))]
            
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        final_cluster_group_counts = [defaultdict(int) for _ in range(k)]

        for group_id in range(M):
            for client_idx in organized_clients[group_id]:
                client_delta, client_size = original_deltas[client_idx], client_datasizes[client_idx]
                raw_q, raw_b, raw_d = np.zeros(k), np.zeros(k), np.zeros(k)

                current_costs = np.zeros(k)
                for j in range(k):
                    if cluster_client_counts[j] > 0:
                        avg = cluster_sum_weighted_deltas[j] / cluster_sum_datasizes[j]
                        current_costs[j] = np.linalg.norm(avg - global_avg_original_delta)**2
                
                for j in range(k):
                    hypo_counts = cluster_client_counts.copy(); hypo_counts[j] += 1
                    hypo_avg = (cluster_sum_weighted_deltas[j] + client_size*client_delta) / (cluster_sum_datasizes[j] + client_size + 1e-9)
                    new_cost = np.linalg.norm(hypo_avg - global_avg_original_delta)**2
                    raw_q[j] = new_cost - current_costs[j]
                    raw_b[j] = np.var(hypo_counts)
                    raw_d[j] = final_cluster_group_counts[j][group_id]

                q_min, q_max = raw_q.min(), raw_q.max()
                norm_q = (raw_q - q_min) / (q_max - q_min + 1e-9)
                b_min, b_max = raw_b.min(), raw_b.max()
                norm_b = (raw_b - b_min) / (b_max - b_min + 1e-9)
                d_min, d_max = raw_d.min(), raw_d.max()
                norm_d = (raw_d - d_min) / (d_max - d_min + 1e-9) if d_max > d_min else np.zeros(k)
                
                total_costs = norm_q + args.balancelambda * norm_b + args.diversitylambda * norm_d
                best_cluster_idx = np.argmin(total_costs)
                cluster_assignments[client_idx] = best_cluster_idx
                cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                cluster_sum_datasizes[best_cluster_idx] += client_size
                cluster_client_counts[best_cluster_idx] += 1
                final_cluster_group_counts[best_cluster_idx][group_id] += 1

        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "greedy_large_grad_seed":
        print("Running Greedy Clustering (Largest Gradient Seeding)...")

        # --- PREPARATION ---
        # We need to modify the standard prep to get un-normalized deltas as well
        print("Preparing deltas and running PCA...")
        all_deltas_normalized = []
        all_deltas_unnormalized = []
        client_datasizes = []
        initial_weight_dict = centralserver.model.state_dict()
        with torch.no_grad():
            initial_weight_flat = torch.cat([p.view(-1) for p in initial_weight_dict.values()]).cpu()

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
            all_deltas_unnormalized.append(delta_np) # Save the raw delta
            
            row_norm = np.linalg.norm(delta_np)
            normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
            all_deltas_normalized.append(normalized_delta)
        print() 

        original_deltas = np.array(all_deltas_unnormalized)
        client_datasizes = np.array(client_datasizes)
        N_COMPONENTS = 100
        pca = PCA(n_components=N_COMPONENTS)
        reduced_deltas = pca.fit_transform(np.array(all_deltas_normalized))
        print(f"PCA complete, using {N_COMPONENTS} components.")
        
        k = args.clusternum
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])

        # --- STEP 1: KMeans on Gradients ---
        print("STEP 1: Pre-clustering on gradients...")
        M = k 
        kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        initial_groups = kmeans.labels_ 

        # --- STEP 2: Largest Gradient Seeding Phase ---
        print("STEP 2: Seeding clusters with the largest gradient client from each group...")
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        
        remaining_client_indices = list(range(num_clients))
        
        for group_id in range(k):
            indices_in_group = np.where(initial_groups == group_id)[0]
            if len(indices_in_group) == 0:
                continue
            
            # Find the client with the largest gradient norm in this group
            group_deltas = original_deltas[indices_in_group]
            group_norms = np.linalg.norm(group_deltas, axis=1)
            largest_grad_relative_idx = np.argmax(group_norms)
            largest_grad_original_idx = indices_in_group[largest_grad_relative_idx]
            
            # Assign this client to a unique cluster
            final_cluster_id = group_id
            cluster_assignments[largest_grad_original_idx] = final_cluster_id
            
            # Update the state of the new cluster
            client_delta = original_deltas[largest_grad_original_idx]
            client_size = client_datasizes[largest_grad_original_idx]
            
            cluster_sum_weighted_deltas[final_cluster_id] += client_size * client_delta
            cluster_sum_datasizes[final_cluster_id] += client_size
            cluster_client_counts[final_cluster_id] += 1
            
            remaining_client_indices.remove(largest_grad_original_idx)

        # --- STEP 3: Greedy Assignment for Remaining Clients ---
        print("STEP 3: Assigning remaining clients greedily...")

        organized_remaining_clients = [[] for _ in range(k)]
        for client_idx in remaining_client_indices:
            group_id = initial_groups[client_idx]
            organized_remaining_clients[group_id].append(client_idx)

        # 3. Process remaining clients using a round-robin order
        max_group_size = max(len(g) for g in organized_remaining_clients) if organized_remaining_clients else 0
        for i in range(max_group_size):
            for group_id in range(k):
                if i < len(organized_remaining_clients[group_id]):
                    client_idx = organized_remaining_clients[group_id][i]
                    
                    client_delta = original_deltas[client_idx]
                    client_size = client_datasizes[client_idx]
                    
                    raw_quality_scores = np.zeros(k)

                    for j in range(k):
                        hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                        
                        # 4. Score: Global Variance of ORIGINAL Gradients
                        hypothetical_cluster_avgs = []
                        for cid in range(k):
                            if hypothetical_counts[cid] > 0:
                                sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                                avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                                hypothetical_cluster_avgs.append(avg)
                        raw_quality_scores[j] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0

                    best_cluster_idx = np.argmin(raw_quality_scores)
                    cluster_assignments[client_idx] = best_cluster_idx
                    
                    # Permanently update the state
                    cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                    cluster_sum_datasizes[best_cluster_idx] += client_size
                    cluster_client_counts[best_cluster_idx] += 1
                    
        print("\nLargest-gradient-seeded greedy assignment complete.")
        print("Final cluster client counts:", cluster_client_counts)
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)
    


    

    elif args.clusteringtype == "clusterbysystemsimilarity":

        print("Clustering clients by system similarity (training time)...")
        
        # Prepare for evaluation
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        
        # --- FIXED: Calculate training times once ---
        # Calculate the training time for each client and store it as a NumPy array
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        
        # The input for KMeans needs to be 2D, so reshape it
        client_training_times_2d = client_training_times.reshape(-1, 1)
        
        # Clustering, kmeans based on training time
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(client_training_times_2d)
        print("Cluster assignments based on training time:", cluster_assignments)
        
        # Evaluation by variance
        # Pass the original 1D array of training times to the evaluation function
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    # elif args.clusteringtype == "clusterbygradientdissimilaritysystemsimilarity":

    #     k = args.clusternum
    #     num_clients = len(clientlist)
    #     cluster_assignments = None
    #     client_training_times = np.array([client.calculate_training_time() for client in clientlist])

    #     print("Running a Multi-Objective HYBRID clustering...")
    #     reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
    #     N_COMPONENTS = reduced_deltas.shape[1]

    #     M = k 
    #     kmeans = KMeans(n_clusters=M, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
    #     initial_groups, group_centroids = kmeans.labels_, kmeans.cluster_centers_
        
    #     organized_clients = [[] for _ in range(M)]
    #     for client_idx, group_id in enumerate(initial_groups): organized_clients[group_id].append(client_idx)
    #     for group_id in range(M):
    #         clients_in_group = organized_clients[group_id]
    #         if not clients_in_group: continue
    #         centroid = group_centroids[group_id]
    #         distances = [np.linalg.norm(reduced_deltas[c_idx] - centroid) for c_idx in clients_in_group]
    #         organized_clients[group_id] = [c_idx for _, c_idx in sorted(zip(distances, clients_in_group))]

    #     global_avg_reduced_delta = np.average(reduced_deltas, axis=0, weights=client_datasizes)
    #     cluster_assignments = np.full(num_clients, -1, dtype=int)
    #     cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS))
    #     cluster_sum_datasizes = np.zeros(k)
    #     cluster_client_counts = np.zeros(k, dtype=int)
    #     final_cluster_group_counts = [defaultdict(int) for _ in range(k)]
    #     cluster_training_times_list = [[] for _ in range(k)]
        
    #     max_group_size = max(len(g) for g in organized_clients) if organized_clients else 0
    #     for i in range(max_group_size):
    #         for group_id in range(M):
    #             if i < len(organized_clients[group_id]):
    #                 client_idx = organized_clients[group_id][i]
    #                 client_delta, client_size, client_time = reduced_deltas[client_idx], client_datasizes[client_idx], client_training_times[client_idx]
                    
    #                 raw_q, raw_b, raw_d, raw_s = np.zeros(k), np.zeros(k), np.zeros(k), np.zeros(k)
                    
    #                 current_cluster_quality_costs = np.zeros(k)
    #                 for j in range(k):
    #                     if cluster_client_counts[j] > 0:
    #                         current_avg = cluster_sum_weighted_deltas[j] / cluster_sum_datasizes[j]
    #                         current_cluster_quality_costs[j] = np.linalg.norm(current_avg - global_avg_reduced_delta)**2

    #                 for j in range(k):
    #                     hypothetical_avg = (cluster_sum_weighted_deltas[j] + client_size * client_delta) / (cluster_sum_datasizes[j] + client_size + 1e-9)
    #                     new_quality_cost = np.linalg.norm(hypothetical_avg - global_avg_reduced_delta)**2
    #                     raw_q[j] = new_quality_cost - current_cluster_quality_costs[j]
    #                     raw_b[j] = cluster_client_counts[j]
    #                     raw_d[j] = final_cluster_group_counts[j][group_id]
    #                     hypothetical_times = cluster_training_times_list[j] + [client_time]
    #                     raw_s[j] = np.var(hypothetical_times)

    #                 norm_q = (raw_q - raw_q.min()) / (raw_q.max() - raw_q.min() + 1e-9)
    #                 norm_b = (raw_b - raw_b.min()) / (raw_b.max() - raw_b.min() + 1e-9)
    #                 norm_d = (raw_d - raw_d.min()) / (raw_d.max() - raw_d.min() + 1e-9) if raw_d.max() > raw_d.min() else np.zeros(k)
    #                 norm_s = (raw_s - raw_s.min()) / (raw_s.max() - raw_s.min() + 1e-9)

    #                 total_costs = norm_q + args.balancelambda * norm_b + args.diversitylambda * norm_d + args.systemlambda * norm_s
    #                 best_cluster_idx = np.argmin(total_costs)
    #                 cluster_assignments[client_idx] = best_cluster_idx
                    
    #                 cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
    #                 cluster_sum_datasizes[best_cluster_idx] += client_size
    #                 cluster_client_counts[best_cluster_idx] += 1
    #                 final_cluster_group_counts[best_cluster_idx][group_id] += 1
    #                 cluster_training_times_list[best_cluster_idx].append(client_time)
        
    #     evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)





    elif args.clusteringtype == "clusterbygradientdissimilaritysystemsimilarity":
        print("Running Greedy Clustering (Straggler Seeding, Global Variance)...")

        # --- PREPARATION ---
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])

        # --- STEP 1: Pre-cluster on Training Time to define groups ---
        print("STEP 1: Pre-clustering on system time...")
        kmeans_system = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        system_labels = kmeans_system.fit_predict(client_training_times.reshape(-1, 1))

        # --- STEP 2: Straggler Seeding Phase ---
        print("STEP 2: Seeding clusters with the slowest client from each system group...")
        
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        cluster_training_times_list = [[] for _ in range(k)]
        
        remaining_client_indices = list(range(num_clients))
        
        for system_group_id in range(k):
            indices_in_group = np.where(system_labels == system_group_id)[0]
            if len(indices_in_group) == 0:
                continue
            
            group_times = client_training_times[indices_in_group]
            straggler_relative_idx = np.argmax(group_times)
            straggler_original_idx = indices_in_group[straggler_relative_idx]
            
            final_cluster_id = system_group_id
            cluster_assignments[straggler_original_idx] = final_cluster_id
            
            client_delta = original_deltas[straggler_original_idx]
            client_size = client_datasizes[straggler_original_idx]
            client_time = client_training_times[straggler_original_idx]
            
            cluster_sum_weighted_deltas[final_cluster_id] += client_size * client_delta
            cluster_sum_datasizes[final_cluster_id] += client_size
            cluster_client_counts[final_cluster_id] += 1
            cluster_training_times_list[final_cluster_id].append(client_time)
            
            remaining_client_indices.remove(straggler_original_idx)

        # --- STEP 3: Greedy Assignment for Remaining Clients ---
        print("STEP 3: Assigning remaining clients greedily...")

        organized_remaining_clients = [[] for _ in range(k)]
        for client_idx in remaining_client_indices:
            system_id = system_labels[client_idx]
            organized_remaining_clients[system_id].append(client_idx)

        max_group_size = max(len(g) for g in organized_remaining_clients) if organized_remaining_clients else 0
        for i in range(max_group_size):
            for system_group_id in range(k):
                if i < len(organized_remaining_clients[system_group_id]):
                    client_idx = organized_remaining_clients[system_group_id][i]
                    
                    client_delta = original_deltas[client_idx]
                    client_size = client_datasizes[client_idx]
                    client_time = client_training_times[client_idx]
                    
                    raw_quality_scores = np.zeros(k)
                    raw_system_scores = np.zeros(k)

                    for j in range(k):
                        hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                        
                        # 1. Quality Score: Global Variance of ORIGINAL Gradients
                        hypothetical_cluster_avgs = []
                        for cid in range(k):
                            if hypothetical_counts[cid] > 0:
                                sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                                avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                                hypothetical_cluster_avgs.append(avg)
                        raw_quality_scores[j] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0

                        # --- MODIFIED: System Score is now Mean of Straggler Times ---
                        hypothetical_straggler_times = np.zeros(k)
                        for cid in range(k):
                            times = cluster_training_times_list[cid]
                            if cid == j:
                                times = times + [client_time]
                            
                            if times: # Check if the list is not empty
                                hypothetical_straggler_times[cid] = np.max(times)
                        
                        hypothetical_non_empty_times = [t for t in hypothetical_straggler_times if t > 0]
                        raw_system_scores[j] = np.mean(hypothetical_non_empty_times) if hypothetical_non_empty_times else 0.0
                        # --- End of Modification ---

                    # Normalize scores using Min-Max Scaling
                    q_min, q_max = raw_quality_scores.min(), raw_quality_scores.max()
                    norm_q = (raw_quality_scores - q_min) / (q_max - q_min + 1e-9)

                    s_min, s_max = raw_system_scores.min(), raw_system_scores.max()
                    norm_s = (raw_system_scores - s_min) / (s_max - s_min + 1e-9)
                    
                    total_costs = norm_q + args.systemlambda * norm_s

                    best_cluster_idx = np.argmin(total_costs)
                    cluster_assignments[client_idx] = best_cluster_idx
                    
                    # Permanently update the state
                    cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                    cluster_sum_datasizes[best_cluster_idx] += client_size
                    cluster_client_counts[best_cluster_idx] += 1
                    cluster_training_times_list[best_cluster_idx].append(client_time)
                    
        print("\nStraggler-seeded greedy assignment complete.")
        print("Final cluster client counts:", cluster_client_counts)
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)

    
    elif args.clusteringtype == "clusterbygradientdissimilaritysystemsimilarityvariance":
        print("Running Greedy Clustering (Straggler Seeding, Mean Time Variance)...")

        # --- PREPARATION ---
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)
        k = args.clusternum
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])

        # --- STEP 1: Pre-cluster on Training Time to define groups ---
        print("STEP 1: Pre-clustering on system time...")
        kmeans_system = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        system_labels = kmeans_system.fit_predict(client_training_times.reshape(-1, 1))

        # --- STEP 2: Straggler Seeding Phase ---
        print("STEP 2: Seeding clusters with the slowest client from each system group...")
        
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        cluster_training_times_list = [[] for _ in range(k)]
        
        remaining_client_indices = list(range(num_clients))
        
        for system_group_id in range(k):
            indices_in_group = np.where(system_labels == system_group_id)[0]
            if len(indices_in_group) == 0:
                continue
            
            group_times = client_training_times[indices_in_group]
            straggler_relative_idx = np.argmax(group_times)
            straggler_original_idx = indices_in_group[straggler_relative_idx]
            
            final_cluster_id = system_group_id
            cluster_assignments[straggler_original_idx] = final_cluster_id
            
            client_delta = original_deltas[straggler_original_idx]
            client_size = client_datasizes[straggler_original_idx]
            client_time = client_training_times[straggler_original_idx]
            
            cluster_sum_weighted_deltas[final_cluster_id] += client_size * client_delta
            cluster_sum_datasizes[final_cluster_id] += client_size
            cluster_client_counts[final_cluster_id] += 1
            cluster_training_times_list[final_cluster_id].append(client_time)
            
            remaining_client_indices.remove(straggler_original_idx)

        # --- STEP 3: Greedy Assignment for Remaining Clients ---
        print("STEP 3: Assigning remaining clients greedily...")

        organized_remaining_clients = [[] for _ in range(k)]
        for client_idx in remaining_client_indices:
            system_id = system_labels[client_idx]
            organized_remaining_clients[system_id].append(client_idx)

        max_group_size = max(len(g) for g in organized_remaining_clients) if organized_remaining_clients else 0
        for i in range(max_group_size):
            for system_group_id in range(k):
                if i < len(organized_remaining_clients[system_group_id]):
                    client_idx = organized_remaining_clients[system_group_id][i]
                    
                    client_delta = original_deltas[client_idx]
                    client_size = client_datasizes[client_idx]
                    client_time = client_training_times[client_idx]
                    
                    raw_quality_scores = np.zeros(k)
                    raw_system_scores = np.zeros(k)

                    for j in range(k):
                        hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                        
                        # 1. Quality Score: Global Variance of ORIGINAL Gradients
                        hypothetical_cluster_avgs = []
                        for cid in range(k):
                            if hypothetical_counts[cid] > 0:
                                sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                                avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                                hypothetical_cluster_avgs.append(avg)
                        raw_quality_scores[j] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0

                        # --- MODIFIED: System Score is now Mean of Intra-Cluster Variances ---
                        hypothetical_variances = []
                        for cid in range(k):
                            times = cluster_training_times_list[cid]
                            if cid == j:
                                times = times + [client_time]
                            # np.var of a single item or empty list is 0
                            hypothetical_variances.append(np.var(times))
                        raw_system_scores[j] = np.mean(hypothetical_variances)
                        # --- End of Modification ---

                    # Normalize scores using Min-Max Scaling
                    q_min, q_max = raw_quality_scores.min(), raw_quality_scores.max()
                    norm_q = (raw_quality_scores - q_min) / (q_max - q_min + 1e-9)

                    s_min, s_max = raw_system_scores.min(), raw_system_scores.max()
                    norm_s = (raw_system_scores - s_min) / (s_max - s_min + 1e-9)
                    
                    total_costs = norm_q + args.systemlambda * norm_s

                    best_cluster_idx = np.argmin(total_costs)
                    cluster_assignments[client_idx] = best_cluster_idx
                    
                    # Permanently update the state
                    cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
                    cluster_sum_datasizes[best_cluster_idx] += client_size
                    cluster_client_counts[best_cluster_idx] += 1
                    cluster_training_times_list[best_cluster_idx].append(client_time)
                    
        print("\nStraggler-seeded greedy assignment complete.")
        print("Final cluster client counts:", cluster_client_counts)
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)





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

def evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times):
    """
    Calculates and prints inter-cluster, intra-cluster, and system straggler metrics.
    """
    print("\n--- Clustering Evaluation Metrics (on Original Gradients) ---")
    
    # --- 1. Inter-Cluster Variance (Gradient Similarity) ---
    print("1. Inter-Cluster Variance (Gradient Similarity):")
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

    if not cluster_averages:
        logging.info("  - N/A (no non-empty clusters)")
    elif len(cluster_averages) == 1:
        logging.info("  - Weighted Variance of Gradients: 0.000000 (only one cluster)")
        logging.info("  - Mean Margin from Global Avg: 0.000000 (only one cluster)")
    else:
        cluster_averages_np = np.array(cluster_averages)
        global_average_gradient = np.average(cluster_averages_np, axis=0, weights=cluster_total_datasizes)
        squared_distances = np.sum((cluster_averages_np - global_average_gradient)**2, axis=1)
        
        weighted_inter_variance = np.average(squared_distances, weights=cluster_total_datasizes)
        logging.info(f"  - Weighted Variance of Gradients: {weighted_inter_variance:.6f} (lower is better)")
        
        mean_margin = np.mean(squared_distances)
        logging.info(f"  - Mean Margin from Global Avg: {mean_margin:.6f} (lower is better)")

    # --- 2. Intra-Cluster Variance (Gradient Similarity) ---
    print("2. Intra-Cluster Variance (Gradient Similarity):")
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

    # --- 3. System Straggler Time ---
    print("3. System Variance (Training Time Similarity):")
    non_empty_cluster_straggler_times = []
    all_cluster_straggler_times = np.zeros(k)
    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 0:
            times_in_cluster = client_training_times[indices_in_cluster]
            max_time = np.max(times_in_cluster)
            all_cluster_straggler_times[j] = max_time
            non_empty_cluster_straggler_times.append(max_time)

    mean_straggler_time = np.mean(non_empty_cluster_straggler_times) if non_empty_cluster_straggler_times else 0.0
    logging.info(f"  - Per-Cluster Straggler Times: {[float(f'{t:.2f}') for t in all_cluster_straggler_times]}")
    logging.info(f"  - Mean Straggler Time (of non-empty clusters): {mean_straggler_time:.2f} (lower is better)")