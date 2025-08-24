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
from scipy.optimize import linear_sum_assignment


def cluster_clients(centralserver, clientlist, args):

    print(f"DEBUG: Received clustering type: '{args.clusteringtype}'")

    # centralserver = CentralServer(args.interclusteringtype, args.centralserverepoch, args, [])

    if args.clusteringtype == "clientorder":

        # prepare for clustering
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
        num_clients = len(clientlist)
        
        # clustering, same cluster size
        cluster_assignments = np.zeros(num_clients, dtype=int)
        clients_per_cluster = (num_clients + k - 1) // k
        for client_idx in range(num_clients):
            cluster_assignments[client_idx] = client_idx // clients_per_cluster

        # evaluate clustering by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "random":
       
        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
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
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "gradientsimilarity":

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
        
        # clustering, kmeans on pca vectors
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto").fit(reduced_deltas)
        cluster_assignments = kmeans.labels_

        # evaluation by variance
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)

    elif args.clusteringtype == "gradientbetween":
        
        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
        level = args.roundrobinlevel
       
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


    elif args.clusteringtype == "systemsimilarity":

        print("Clustering clients by system similarity (training time)...")
        
        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
        
        # The input for KMeans needs to be 2D, so reshape it
        client_training_times_2d = client_training_times.reshape(-1, 1)
        
        # Clustering, kmeans based on training time
        kmeans = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(client_training_times_2d)
        print("Cluster assignments based on training time:", cluster_assignments)
        
        # Evaluation by variance
        # Pass the original 1D array of training times to the evaluation function
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "fedch":
        print("Running Adapted FedCH (Clustering by Representative Time Difference)...")

        # --- PREPARATION ---
        k = args.clusternum
        num_clients = len(clientlist)
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        # We don't need the gradients for this method, but we need the datasizes for evaluation
        _, client_datasizes, original_deltas = prepare_and_run_pca(clientlist, centralserver, args)[:3]
        
        # --- STEP 1: Initialization and Seeding ---
        print("STEP 1: Initializing and seeding clusters...")
        # Randomly select initial representative clients for each cluster
        representative_indices = np.random.choice(num_clients, k, replace=False)
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        
        # Assign each representative to its own cluster
        for i in range(k):
            cluster_assignments[representative_indices[i]] = i
            
        remaining_client_indices = list(set(range(num_clients)) - set(representative_indices))
        
        # --- STEP 2: Initial Assignment of Remaining Clients (Hungarian Algorithm) ---
        print("STEP 2: Performing initial balanced assignment with Hungarian algorithm...")
        
        # Create the slots for the remaining N-k clients
        slots_per_cluster = (num_clients - k) // k
        remaining_slots = (num_clients - k) % k
        
        slot_to_cluster_map = []
        for i in range(k):
            num_slots = slots_per_cluster + (1 if i < remaining_slots else 0)
            slot_to_cluster_map.extend([i] * num_slots)

        # Create the cost matrix for the N-k clients and N-k slots
        cost_matrix = np.zeros((len(remaining_client_indices), len(remaining_client_indices)))
        representative_times = client_training_times[representative_indices]

        for i, client_idx in enumerate(remaining_client_indices):
            for j, cluster_id in enumerate(slot_to_cluster_map):
                cost_matrix[i, j] = abs(client_training_times[client_idx] - representative_times[cluster_id])

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i in range(len(row_ind)):
            client_idx = remaining_client_indices[row_ind[i]]
            slot_idx = col_ind[i]
            cluster_id = slot_to_cluster_map[slot_idx]
            cluster_assignments[client_idx] = cluster_id

        # --- STEP 3: Iterative Refinement Loop ---
        for epoch in range(5): # Iterate for a few epochs to allow convergence
            print(f"  - Refinement Epoch {epoch+1}...")
            
            # --- Update Representatives (Medoids) ---
            new_representative_indices = np.zeros(k, dtype=int)
            for j in range(k):
                indices_in_cluster = np.where(cluster_assignments == j)[0]
                if len(indices_in_cluster) == 0:
                    unassigned = list(set(range(num_clients)) - set(representative_indices))
                    new_representative_indices[j] = np.random.choice(unassigned) if unassigned else representative_indices[j]
                    continue

                min_total_dissimilarity = float('inf')
                best_representative_idx = -1
                
                for potential_rep_idx in indices_in_cluster:
                    total_dissimilarity = 0
                    for other_client_idx in indices_in_cluster:
                        total_dissimilarity += abs(client_training_times[potential_rep_idx] - client_training_times[other_client_idx])
                    
                    if total_dissimilarity < min_total_dissimilarity:
                        min_total_dissimilarity = total_dissimilarity
                        best_representative_idx = potential_rep_idx
                
                new_representative_indices[j] = best_representative_idx
            
            representative_indices = new_representative_indices
            
            # --- Re-run Balanced Assignment for all clients ---
            slots_per_cluster = num_clients // k
            remaining_slots = num_clients % k
            slot_to_cluster_map = []
            for i in range(k):
                num_slots = slots_per_cluster + (1 if i < remaining_slots else 0)
                slot_to_cluster_map.extend([i] * num_slots)
                
            cost_matrix = np.zeros((num_clients, num_clients))
            representative_times = client_training_times[representative_indices]
            
            for i in range(num_clients):
                for j, cluster_id in enumerate(slot_to_cluster_map):
                    cost_matrix[i, j] = abs(client_training_times[i] - representative_times[cluster_id])

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            temp_assignments = np.full(num_clients, -1, dtype=int)
            for i in range(len(row_ind)):
                client_idx = row_ind[i]
                slot_idx = col_ind[i]
                cluster_id = slot_to_cluster_map[slot_idx]
                temp_assignments[client_idx] = cluster_id

            # Check for convergence
            if np.array_equal(temp_assignments, cluster_assignments):
                print("  - Converged.")
                break
            cluster_assignments = temp_assignments

        print("\nAdapted FedCH clustering complete.")
        print("Final cluster client counts:", np.bincount(cluster_assignments))
        
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)




    


    elif args.clusteringtype == "gradientgreedy_normseeding":
        # 1. Seeding: Largest Gradient Norm
        # 2. Score: Mean Cosine Distance
        # 3. Refinement: Yes (5 epochs)
        print("Running Greedy 14: Seed by Norm, Score by Cosine Distance, with Refinement...")

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas, _ = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)

        # --- STEP 1: Seeding Phase ---
        print("STEP 1: Seeding clusters with k largest gradient norm clients...")
        client_gradient_norms = np.linalg.norm(original_deltas, axis=1)
        sorted_clients_by_norm = np.argsort(client_gradient_norms)[::-1]
        
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        global_avg_norm = np.linalg.norm(global_avg_original_delta)
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        
        clients_to_assign = list(range(num_clients))
        
        for i in range(k):
            if not clients_to_assign: break
            client_idx = sorted_clients_by_norm[i]
            cluster_assignments[client_idx] = i
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            cluster_sum_weighted_deltas[i] += client_size * client_delta
            cluster_sum_datasizes[i] += client_size
            cluster_client_counts[i] += 1
            clients_to_assign.remove(client_idx)
            
        # --- STEP 2: Greedy Assignment for Remaining Clients ---
        print("STEP 2: Assigning remaining clients greedily...")
        for client_idx in clients_to_assign:
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            raw_quality_scores = np.zeros(k)

            for j in range(k):
                hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                cos_distances = []
                for cid in range(k):
                    if hypothetical_counts[cid] > 0:
                        sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                        avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                        avg_norm = np.linalg.norm(avg)
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            cos_distances.append(1.0 - cos_sim)
                raw_quality_scores[j] = np.mean(cos_distances) if cos_distances else 0

            best_cluster_idx = np.argmin(raw_quality_scores)
            cluster_assignments[client_idx] = best_cluster_idx
            
            cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
            cluster_sum_datasizes[best_cluster_idx] += client_size
            cluster_client_counts[best_cluster_idx] += 1

        # --- STEP 3: Iterative Refinement ---
        print("STEP 3: Refining assignments for 5 epochs...")
        for epoch in range(5):
            client_moved = False
            for client_idx in range(num_clients):
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                current_cluster_id = cluster_assignments[client_idx]

                # Calculate current global score
                current_cos_distances = []
                for cid in range(k):
                    if cluster_client_counts[cid] > 0:
                        avg = cluster_sum_weighted_deltas[cid] / cluster_sum_datasizes[cid]
                        avg_norm = np.linalg.norm(avg)
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            current_cos_distances.append(1.0 - cos_sim)
                current_score = np.mean(current_cos_distances) if current_cos_distances else 0

                # Temporarily remove client
                cluster_sum_weighted_deltas[current_cluster_id] -= client_size * client_delta
                cluster_sum_datasizes[current_cluster_id] -= client_size
                cluster_client_counts[current_cluster_id] -= 1

                move_scores = {}
                for new_cluster_id in range(k):
                    if new_cluster_id == current_cluster_id: continue
                    
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[new_cluster_id] += 1
                    hypothetical_cos_distances = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == new_cluster_id else sum_d/sum_s
                            avg_norm = np.linalg.norm(avg)
                            if avg_norm > 0 and global_avg_norm > 0:
                                cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                                hypothetical_cos_distances.append(1.0 - cos_sim)
                    move_scores[new_cluster_id] = np.mean(hypothetical_cos_distances) if hypothetical_cos_distances else 0

                best_new_cluster_id = min(move_scores, key=move_scores.get)
                if move_scores[best_new_cluster_id] < current_score:
                    cluster_assignments[client_idx] = best_new_cluster_id
                    cluster_sum_weighted_deltas[best_new_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[best_new_cluster_id] += client_size
                    cluster_client_counts[best_new_cluster_id] += 1
                    client_moved = True
                else:
                    # Put client back
                    cluster_sum_weighted_deltas[current_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[current_cluster_id] += client_size
                    cluster_client_counts[current_cluster_id] += 1
            
            if not client_moved:
                print(f"Refinement converged early at epoch {epoch+1}.")
                break
                
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)



    elif args.clusteringtype == "gradientgreedy_cosineseeding":
        # 1. Seeding: Largest Cosine Distance
        # 2. Score: Mean Cosine Distance
        # 3. Refinement: Yes (5 epochs)
        print("Running Greedy 16: Seed by Cosine Dist, Score by Cosine Distance, with Refinement...")

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas, _ = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)

        # --- STEP 1: Seeding Phase ---
        print("STEP 1: Seeding clusters with k largest cosine distance clients...")
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        global_avg_norm = np.linalg.norm(global_avg_original_delta)
        client_cosine_distances = np.zeros(num_clients)
        for i in range(num_clients):
            client_delta = original_deltas[i]
            client_norm = np.linalg.norm(client_delta)
            if client_norm > 0 and global_avg_norm > 0:
                cos_sim = np.dot(client_delta, global_avg_original_delta) / (client_norm * global_avg_norm)
                client_cosine_distances[i] = 1.0 - cos_sim
        sorted_clients_by_dist = np.argsort(client_cosine_distances)[::-1]
        
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        
        clients_to_assign = list(range(num_clients))
        
        for i in range(k):
            if not clients_to_assign: break
            client_idx = sorted_clients_by_dist[i]
            cluster_assignments[client_idx] = i
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            cluster_sum_weighted_deltas[i] += client_size * client_delta
            cluster_sum_datasizes[i] += client_size
            cluster_client_counts[i] += 1
            clients_to_assign.remove(client_idx)
            
        # --- STEP 2: Greedy Assignment for Remaining Clients ---
        print("STEP 2: Assigning remaining clients greedily...")
        for client_idx in clients_to_assign:
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            raw_quality_scores = np.zeros(k)

            for j in range(k):
                hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                cos_distances = []
                for cid in range(k):
                    if hypothetical_counts[cid] > 0:
                        sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                        avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                        avg_norm = np.linalg.norm(avg)
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            cos_distances.append(1.0 - cos_sim)
                raw_quality_scores[j] = np.mean(cos_distances) if cos_distances else 0

            best_cluster_idx = np.argmin(raw_quality_scores)
            cluster_assignments[client_idx] = best_cluster_idx
            
            cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
            cluster_sum_datasizes[best_cluster_idx] += client_size
            cluster_client_counts[best_cluster_idx] += 1

        # --- STEP 3: Iterative Refinement ---
        print("STEP 3: Refining assignments for 5 epochs...")
        for epoch in range(5):
            client_moved = False
            for client_idx in range(num_clients):
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                current_cluster_id = cluster_assignments[client_idx]

                # Calculate current global score
                current_cluster_avgs = []
                for cid in range(k):
                    if cluster_client_counts[cid] > 0:
                        current_cluster_avgs.append(cluster_sum_weighted_deltas[cid] / cluster_sum_datasizes[cid])
                current_score = np.var(np.array(current_cluster_avgs), axis=0).sum() if len(current_cluster_avgs) > 1 else 0

                # Temporarily remove client
                cluster_sum_weighted_deltas[current_cluster_id] -= client_size * client_delta
                cluster_sum_datasizes[current_cluster_id] -= client_size
                cluster_client_counts[current_cluster_id] -= 1

                move_scores = {}
                for new_cluster_id in range(k):
                    if new_cluster_id == current_cluster_id: continue
                    
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[new_cluster_id] += 1
                    hypothetical_cluster_avgs = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == new_cluster_id else sum_d/sum_s
                            hypothetical_cluster_avgs.append(avg)
                    move_scores[new_cluster_id] = np.var(np.array(hypothetical_cluster_avgs), axis=0).sum() if len(hypothetical_cluster_avgs) > 1 else 0

                best_new_cluster_id = min(move_scores, key=move_scores.get)
                if move_scores[best_new_cluster_id] < current_score:
                    cluster_assignments[client_idx] = best_new_cluster_id
                    cluster_sum_weighted_deltas[best_new_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[best_new_cluster_id] += client_size
                    cluster_client_counts[best_new_cluster_id] += 1
                    client_moved = True
                else:
                    # Put client back
                    cluster_sum_weighted_deltas[current_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[current_cluster_id] += client_size
                    cluster_client_counts[current_cluster_id] += 1
            
            if not client_moved:
                print(f"Refinement converged early at epoch {epoch+1}.")
                break
                
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "gradientgreedy":
        # 1. Seeding: K random clients
        # 2. Score: Mean Cosine Distance
        # 3. Refinement: Yes (5 epochs, random order)
        print("Running Greedy with Random Seeding, Score by Cosine Distance, with Refinement...")

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas, _ = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)

        # --- STEP 1: Seeding Phase (MODIFIED) ---
        print("STEP 1: Seeding clusters with k random clients...")
        
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        global_avg_norm = np.linalg.norm(global_avg_original_delta)
        cluster_assignments = np.full(num_clients, -1, dtype=int)
        cluster_sum_weighted_deltas = np.zeros((k, N_COMPONENTS_ORIGINAL))
        cluster_sum_datasizes = np.zeros(k)
        cluster_client_counts = np.zeros(k, dtype=int)
        
        clients_to_assign = list(range(num_clients))
        random.shuffle(clients_to_assign) # Shuffle clients for random selection
        
        for i in range(k):
            if not clients_to_assign: break
            # Pop the next random client from the shuffled list
            client_idx = clients_to_assign.pop(0) 
            
            cluster_assignments[client_idx] = i
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            cluster_sum_weighted_deltas[i] += client_size * client_delta
            cluster_sum_datasizes[i] += client_size
            cluster_client_counts[i] += 1
            
        # --- STEP 2: Greedy Assignment for Remaining Clients ---
        print("STEP 2: Assigning remaining clients greedily...")
        # The remaining clients in clients_to_assign are already in a random order
        for client_idx in clients_to_assign:
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            raw_quality_scores = np.zeros(k)

            for j in range(k):
                hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                cos_distances = []
                for cid in range(k):
                    if hypothetical_counts[cid] > 0:
                        sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                        avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                        avg_norm = np.linalg.norm(avg)
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            cos_distances.append(1.0 - cos_sim)
                raw_quality_scores[j] = np.mean(cos_distances) if cos_distances else 0

            best_cluster_idx = np.argmin(raw_quality_scores)
            cluster_assignments[client_idx] = best_cluster_idx
            
            cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
            cluster_sum_datasizes[best_cluster_idx] += client_size
            cluster_client_counts[best_cluster_idx] += 1

        # --- STEP 3: Iterative Refinement (MODIFIED) ---
        print("STEP 3: Refining assignments for 5 epochs...")
        for epoch in range(20):
            client_moved = False
            # Create a new random order for processing clients in each epoch
            refinement_order = list(range(num_clients))
            random.shuffle(refinement_order)

            for client_idx in refinement_order:
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                current_cluster_id = cluster_assignments[client_idx]

                # Calculate current global score
                current_cos_distances = []
                for cid in range(k):
                    if cluster_client_counts[cid] > 0:
                        avg = cluster_sum_weighted_deltas[cid] / cluster_sum_datasizes[cid]
                        avg_norm = np.linalg.norm(avg)
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            current_cos_distances.append(1.0 - cos_sim)
                current_score = np.mean(current_cos_distances) if current_cos_distances else 0

                # Temporarily remove client
                cluster_sum_weighted_deltas[current_cluster_id] -= client_size * client_delta
                cluster_sum_datasizes[current_cluster_id] -= client_size
                cluster_client_counts[current_cluster_id] -= 1

                move_scores = {}
                for new_cluster_id in range(k):
                    if new_cluster_id == current_cluster_id: continue
                    
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[new_cluster_id] += 1
                    hypothetical_cos_distances = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == new_cluster_id else sum_d/sum_s
                            avg_norm = np.linalg.norm(avg)
                            if avg_norm > 0 and global_avg_norm > 0:
                                cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                                hypothetical_cos_distances.append(1.0 - cos_sim)
                    move_scores[new_cluster_id] = np.mean(hypothetical_cos_distances) if hypothetical_cos_distances else 0

                best_new_cluster_id = min(move_scores, key=move_scores.get)
                if move_scores[best_new_cluster_id] < current_score:
                    cluster_assignments[client_idx] = best_new_cluster_id
                    cluster_sum_weighted_deltas[best_new_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[best_new_cluster_id] += client_size
                    cluster_client_counts[best_new_cluster_id] += 1
                    client_moved = True
                else:
                    # Put client back
                    cluster_sum_weighted_deltas[current_cluster_id] += client_size * client_delta
                    cluster_sum_datasizes[current_cluster_id] += client_size
                    cluster_client_counts[current_cluster_id] += 1
            
            if not client_moved:
                print(f"Refinement converged early at epoch {epoch+1}.")
                break
                
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)


    elif args.clusteringtype == "bothgreedy":
        # 1. Seeding: Stragglers from System Time KMeans
        # 2. Score (Quality): Mean Cosine Distance
        # 3. Score (System): Sum of Straggling Gaps
        # 4. Refinement: Yes (5 epochs)
        print("Running Greedy System 4: Score by Cosine Dist + Straggler Gap, with Refinement...")

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas, _ = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)

        # --- STEP 1: Straggler Seeding Phase ---
        print("STEP 1: Seeding clusters with the slowest client from each system group...")
        kmeans_system = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        system_labels = kmeans_system.fit_predict(client_training_times.reshape(-1, 1))
        
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
            straggler_original_idx = indices_in_group[np.argmax(group_times)]
            
            cluster_assignments[straggler_original_idx] = system_group_id
            client_delta = original_deltas[straggler_original_idx]
            client_size = client_datasizes[straggler_original_idx]
            client_time = client_training_times[straggler_original_idx]
            
            cluster_sum_weighted_deltas[system_group_id] += client_size * client_delta
            cluster_sum_datasizes[system_group_id] += client_size
            cluster_client_counts[system_group_id] += 1
            cluster_training_times_list[system_group_id].append(client_time)
            
            remaining_client_indices.remove(straggler_original_idx)

        # --- STEP 2: Greedy Assignment for Remaining Clients ---
        print("STEP 2: Assigning remaining clients greedily...")
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        global_avg_norm = np.linalg.norm(global_avg_original_delta)
        
        for client_idx in remaining_client_indices:
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            client_time = client_training_times[client_idx]
            
            raw_quality_scores = np.zeros(k)
            raw_system_scores = np.zeros(k)

            for j in range(k):
                hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                
                # Quality Score: Mean Cosine Distance (Vectorized)
                hypothetical_avgs_list = []
                for cid in range(k):
                    if hypothetical_counts[cid] > 0:
                        sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                        avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                        hypothetical_avgs_list.append(avg)
                
                if hypothetical_avgs_list:
                    hypothetical_avgs_matrix = np.array(hypothetical_avgs_list)
                    dot_products = np.dot(hypothetical_avgs_matrix, global_avg_original_delta)
                    avg_norms = np.linalg.norm(hypothetical_avgs_matrix, axis=1)
                    cosine_similarities = dot_products / ((avg_norms * global_avg_norm) + 1e-9)
                    raw_quality_scores[j] = np.mean(1.0 - cosine_similarities)
                else:
                    raw_quality_scores[j] = 0

                # System Score: Sum of Straggling Gaps
                hypothetical_gaps = []
                for cid in range(k):
                    times = cluster_training_times_list[cid]
                    if cid == j:
                        times = times + [client_time]
                    if len(times) > 1:
                        gap = np.max(times) - np.mean(times)
                        hypothetical_gaps.append(gap)
                raw_system_scores[j] = np.sum(hypothetical_gaps)

            # Normalize and combine scores
            norm_q = (raw_quality_scores - raw_quality_scores.min()) / (raw_quality_scores.max() - raw_quality_scores.min() + 1e-9)
            norm_s = (raw_system_scores - raw_system_scores.min()) / (raw_system_scores.max() - raw_system_scores.min() + 1e-9)
            total_costs = norm_q + args.systemlambda * norm_s

            best_cluster_idx = np.argmin(total_costs)
            
            print(f"\n--- Assigning Client {client_idx} ---")
            print(f"Raw Quality Scores : {np.round(raw_quality_scores, 4)}")
            print(f"Raw System Scores  : {np.round(raw_system_scores, 4)}")
            print(f"Norm Quality Scores: {np.round(norm_q, 4)}")
            print(f"Norm System Scores : {np.round(norm_s, 4)}")
            print(f"Total Costs        : {np.round(total_costs, 4)}")
            print(f"  >> Assigning to Cluster {best_cluster_idx}")
            
            cluster_assignments[client_idx] = best_cluster_idx
            
            cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
            cluster_sum_datasizes[best_cluster_idx] += client_size
            cluster_client_counts[best_cluster_idx] += 1
            cluster_training_times_list[best_cluster_idx].append(client_time)

        # --- STEP 3: Iterative Refinement ---
        print("STEP 3: Refining assignments for 5 epochs...")
        for epoch in range(5):
            client_moved = False
            refinement_order = list(range(num_clients))
            random.shuffle(refinement_order)

            for client_idx in refinement_order:
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                client_time = client_training_times[client_idx]
                current_cluster_id = cluster_assignments[client_idx]

                # Temporarily remove client from its current cluster
                cluster_sum_weighted_deltas[current_cluster_id] -= client_size * client_delta
                cluster_sum_datasizes[current_cluster_id] -= client_size
                cluster_client_counts[current_cluster_id] -= 1
                cluster_training_times_list[current_cluster_id].remove(client_time)

                raw_quality_scores = np.zeros(k)
                raw_system_scores = np.zeros(k)

                for j in range(k):
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                    
                    # Quality Score
                    hypothetical_avgs_list = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                            hypothetical_avgs_list.append(avg)
                    
                    if hypothetical_avgs_list:
                        hypothetical_avgs_matrix = np.array(hypothetical_avgs_list)
                        dot_products = np.dot(hypothetical_avgs_matrix, global_avg_original_delta)
                        avg_norms = np.linalg.norm(hypothetical_avgs_matrix, axis=1)
                        cosine_similarities = dot_products / ((avg_norms * global_avg_norm) + 1e-9)
                        raw_quality_scores[j] = np.mean(1.0 - cosine_similarities)
                    else:
                        raw_quality_scores[j] = 0

                    # System Score
                    hypothetical_gaps = []
                    for cid in range(k):
                        times = cluster_training_times_list[cid]
                        if cid == j:
                            times = times + [client_time]
                        if len(times) > 1:
                            hypothetical_gaps.append(np.max(times) - np.mean(times))
                    raw_system_scores[j] = np.sum(hypothetical_gaps)

                norm_q = (raw_quality_scores - raw_quality_scores.min()) / (raw_quality_scores.max() - raw_quality_scores.min() + 1e-9)
                norm_s = (raw_system_scores - raw_system_scores.min()) / (raw_system_scores.max() - raw_system_scores.min() + 1e-9)
                total_costs = norm_q + args.systemlambda * norm_s
                
                best_new_cluster_id = np.argmin(total_costs)
                
                print(f"\n--- Refining Client {client_idx} (Current Cluster: {current_cluster_id}) ---")
                print(f"Raw Quality Scores : {np.round(raw_quality_scores, 4)}")
                print(f"Raw System Scores  : {np.round(raw_system_scores, 4)}")
                print(f"Norm Quality Scores: {np.round(norm_q, 4)}")
                print(f"Norm System Scores : {np.round(norm_s, 4)}")
                print(f"Total Costs        : {np.round(total_costs, 4)}")
                
                if best_new_cluster_id != current_cluster_id:
                    print(f"  >> MOVING to new best Cluster {best_new_cluster_id}")
                    cluster_assignments[client_idx] = best_new_cluster_id
                    client_moved = True
                else:
                    print(f"  >> STAYING in Cluster {current_cluster_id}")

                final_assignment = cluster_assignments[client_idx]
                cluster_sum_weighted_deltas[final_assignment] += client_size * client_delta
                cluster_sum_datasizes[final_assignment] += client_size
                cluster_client_counts[final_assignment] += 1
                cluster_training_times_list[final_assignment].append(client_time)
            
            if not client_moved:
                print(f"\nRefinement converged early at epoch {epoch+1}.")
                break
                
        evaluate_clustering(k, cluster_assignments, original_deltas, client_datasizes, client_training_times)
    

    elif args.clusteringtype == "weightedbothgreedy":
        # 1. Seeding: Stragglers from System Time KMeans
        # 2. Score (Quality): Weighted Mean Cosine Distance (penalized by cluster time)
        # 3. Score (System): Sum of Straggling Gaps
        # 4. Refinement: Yes (5 epochs)
        print("Running Greedy System 5: Score by Weighted Cosine Dist + Straggler Gap, with Refinement...")

        # --- PREPARATION ---
        k = args.clusternum
        client_training_times = np.array([client.calculate_training_time() for client in clientlist])
        reduced_deltas, client_datasizes, original_deltas, _ = prepare_and_run_pca(clientlist, centralserver, args)
        N_COMPONENTS_ORIGINAL = original_deltas.shape[1]
        num_clients = len(clientlist)

        # --- STEP 1: Straggler Seeding Phase ---
        print("STEP 1: Seeding clusters with the slowest client from each system group...")
        kmeans_system = KMeans(n_clusters=k, random_state=args.randomseed, n_init="auto")
        system_labels = kmeans_system.fit_predict(client_training_times.reshape(-1, 1))
        
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
            straggler_original_idx = indices_in_group[np.argmax(group_times)]
            
            cluster_assignments[straggler_original_idx] = system_group_id
            client_delta = original_deltas[straggler_original_idx]
            client_size = client_datasizes[straggler_original_idx]
            client_time = client_training_times[straggler_original_idx]
            
            cluster_sum_weighted_deltas[system_group_id] += client_size * client_delta
            cluster_sum_datasizes[system_group_id] += client_size
            cluster_client_counts[system_group_id] += 1
            cluster_training_times_list[system_group_id].append(client_time)
            
            remaining_client_indices.remove(straggler_original_idx)

        # --- STEP 2: Greedy Assignment for Remaining Clients ---
        print("STEP 2: Assigning remaining clients greedily...")
        global_avg_original_delta = np.average(original_deltas, axis=0, weights=client_datasizes)
        global_avg_norm = np.linalg.norm(global_avg_original_delta)
        
        for client_idx in remaining_client_indices:
            client_delta = original_deltas[client_idx]
            client_size = client_datasizes[client_idx]
            client_time = client_training_times[client_idx]
            
            raw_quality_scores = np.zeros(k)
            raw_system_scores = np.zeros(k)

            for j in range(k):
                hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                
                # --- MODIFIED: Weighted Quality Score ---
                weighted_margins = []
                for cid in range(k):
                    if hypothetical_counts[cid] > 0:
                        sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                        avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                        avg_norm = np.linalg.norm(avg)
                        
                        cos_sim = 0.0
                        if avg_norm > 0 and global_avg_norm > 0:
                            cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                        
                        margin = 1.0 - cos_sim

                        times = cluster_training_times_list[cid]
                        if cid == j:
                            times = times + [client_time]
                        
                        straggler_time = np.max(times) if times else 0
                        cluster_total_time = straggler_time + (args.clustercommunicationtime * 2)
                        
                        if cluster_total_time > 0:
                            weighted_margins.append(margin / cluster_total_time)

                raw_quality_scores[j] = np.mean(weighted_margins) if weighted_margins else 0
                # --- End of Modification ---

                # System Score: Sum of Straggling Gaps
                hypothetical_gaps = []
                for cid in range(k):
                    times = cluster_training_times_list[cid]
                    if cid == j:
                        times = times + [client_time]
                    if len(times) > 1:
                        gap = np.max(times) - np.mean(times)
                        hypothetical_gaps.append(gap)
                raw_system_scores[j] = np.sum(hypothetical_gaps)

            # Normalize and combine scores
            norm_q = (raw_quality_scores - raw_quality_scores.min()) / (raw_quality_scores.max() - raw_quality_scores.min() + 1e-9)
            norm_s = (raw_system_scores - raw_system_scores.min()) / (raw_system_scores.max() - raw_system_scores.min() + 1e-9)
            total_costs = norm_q + args.systemlambda * norm_s

            best_cluster_idx = np.argmin(total_costs)
            
            print(f"\n--- Assigning Client {client_idx} ---")
            print(f"Raw Quality Scores : {np.round(raw_quality_scores, 6)}") # Increased precision for new score
            print(f"Raw System Scores  : {np.round(raw_system_scores, 4)}")
            print(f"Norm Quality Scores: {np.round(norm_q, 4)}")
            print(f"Norm System Scores : {np.round(norm_s, 4)}")
            print(f"Total Costs        : {np.round(total_costs, 4)}")
            print(f"  >> Assigning to Cluster {best_cluster_idx}")
            
            cluster_assignments[client_idx] = best_cluster_idx
            
            cluster_sum_weighted_deltas[best_cluster_idx] += client_size * client_delta
            cluster_sum_datasizes[best_cluster_idx] += client_size
            cluster_client_counts[best_cluster_idx] += 1
            cluster_training_times_list[best_cluster_idx].append(client_time)

        # --- STEP 3: Iterative Refinement ---
        print("STEP 3: Refining assignments for 5 epochs...")
        for epoch in range(5):
            client_moved = False
            refinement_order = list(range(num_clients))
            random.shuffle(refinement_order)

            for client_idx in refinement_order:
                client_delta = original_deltas[client_idx]
                client_size = client_datasizes[client_idx]
                client_time = client_training_times[client_idx]
                current_cluster_id = cluster_assignments[client_idx]

                # Temporarily remove client from its current cluster
                cluster_sum_weighted_deltas[current_cluster_id] -= client_size * client_delta
                cluster_sum_datasizes[current_cluster_id] -= client_size
                cluster_client_counts[current_cluster_id] -= 1
                cluster_training_times_list[current_cluster_id].remove(client_time)

                raw_quality_scores = np.zeros(k)
                raw_system_scores = np.zeros(k)

                for j in range(k):
                    hypothetical_counts = cluster_client_counts.copy(); hypothetical_counts[j] += 1
                    
                    # --- MODIFIED: Weighted Quality Score ---
                    weighted_margins = []
                    for cid in range(k):
                        if hypothetical_counts[cid] > 0:
                            sum_d, sum_s = cluster_sum_weighted_deltas[cid], cluster_sum_datasizes[cid]
                            avg = (sum_d + (client_size*client_delta))/(sum_s + client_size) if cid == j else sum_d/sum_s
                            avg_norm = np.linalg.norm(avg)
                            
                            cos_sim = 0.0
                            if avg_norm > 0 and global_avg_norm > 0:
                                cos_sim = np.dot(avg, global_avg_original_delta) / (avg_norm * global_avg_norm)
                            
                            margin = 1.0 - cos_sim

                            times = cluster_training_times_list[cid]
                            if cid == j:
                                times = times + [client_time]
                            
                            straggler_time = np.max(times) if times else 0
                            cluster_total_time = straggler_time + (args.clustercommunicationtime * 2)
                            
                            if cluster_total_time > 0:
                                weighted_margins.append(margin / cluster_total_time)

                    raw_quality_scores[j] = np.mean(weighted_margins) if weighted_margins else 0
                    # --- End of Modification ---

                    # System Score
                    hypothetical_gaps = []
                    for cid in range(k):
                        times = cluster_training_times_list[cid]
                        if cid == j:
                            times = times + [client_time]
                        if len(times) > 1:
                            hypothetical_gaps.append(np.max(times) - np.mean(times))
                    raw_system_scores[j] = np.sum(hypothetical_gaps)

                norm_q = (raw_quality_scores - raw_quality_scores.min()) / (raw_quality_scores.max() - raw_quality_scores.min() + 1e-9)
                norm_s = (raw_system_scores - raw_system_scores.min()) / (raw_system_scores.max() - raw_system_scores.min() + 1e-9)
                total_costs = norm_q + args.systemlambda * norm_s
                
                best_new_cluster_id = np.argmin(total_costs)
                
                print(f"\n--- Refining Client {client_idx} (Current Cluster: {current_cluster_id}) ---")
                print(f"Raw Quality Scores : {np.round(raw_quality_scores, 6)}")
                print(f"Raw System Scores  : {np.round(raw_system_scores, 4)}")
                print(f"Norm Quality Scores: {np.round(norm_q, 4)}")
                print(f"Norm System Scores : {np.round(norm_s, 4)}")
                print(f"Total Costs        : {np.round(total_costs, 4)}")
                
                if best_new_cluster_id != current_cluster_id:
                    print(f"  >> MOVING to new best Cluster {best_new_cluster_id}")
                    cluster_assignments[client_idx] = best_new_cluster_id
                    client_moved = True
                else:
                    print(f"  >> STAYING in Cluster {current_cluster_id}")

                final_assignment = cluster_assignments[client_idx]
                cluster_sum_weighted_deltas[final_assignment] += client_size * client_delta
                cluster_sum_datasizes[final_assignment] += client_size
                cluster_client_counts[final_assignment] += 1
                cluster_training_times_list[final_assignment].append(client_time)
            
            if not client_moved:
                print(f"\nRefinement converged early at epoch {epoch+1}.")
                break
                
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
    then runs PCA. Returns reduced, datasizes, normalized, and original deltas.
    """
    print("Preparing deltas and running PCA...")
    all_deltas_normalized = []
    all_deltas_unnormalized = [] # Added for raw gradients
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
        all_deltas_unnormalized.append(delta_np) # Save the raw delta
        
        row_norm = np.linalg.norm(delta_np)
        normalized_delta = delta_np / (row_norm if row_norm > 0 else 1.0)
        all_deltas_normalized.append(normalized_delta)
    print() 

    all_deltas_normalized_np = np.array(all_deltas_normalized)
    all_deltas_unnormalized_np = np.array(all_deltas_unnormalized)
    client_datasizes_np = np.array(client_datasizes)
    N_COMPONENTS = 100
    pca = PCA(n_components=N_COMPONENTS)
    reduced_deltas = pca.fit_transform(all_deltas_normalized_np)
    print(f"PCA complete, using {N_COMPONENTS} components.")
    
    # Return all four arrays
    return reduced_deltas, client_datasizes_np, all_deltas_unnormalized_np, all_deltas_normalized_np

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
        logging.info(f"  - Per-Cluster Margins: {[float(f'{d:.6f}') for d in squared_distances]}")

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
    total_straggling_gap = 0

    for j in range(k):
        indices_in_cluster = np.where(cluster_assignments == j)[0]
        if len(indices_in_cluster) > 0:
            times_in_cluster = client_training_times[indices_in_cluster]
            max_time = np.max(times_in_cluster)
            all_cluster_straggler_times[j] = max_time
            non_empty_cluster_straggler_times.append(max_time)

            gap = np.sum(max_time - times_in_cluster)
            total_straggling_gap += gap

    mean_straggler_time = np.mean(non_empty_cluster_straggler_times) if non_empty_cluster_straggler_times else 0.0
    logging.info(f"  - Per-Cluster Straggler Times: {[float(f'{t:.2f}') for t in all_cluster_straggler_times]}")
    logging.info(f"  - Mean Straggler Time (of non-empty clusters): {mean_straggler_time:.2f} (lower is better)")
    logging.info(f"  - Total Straggling Gap: {total_straggling_gap:.2f} (lower is better)")