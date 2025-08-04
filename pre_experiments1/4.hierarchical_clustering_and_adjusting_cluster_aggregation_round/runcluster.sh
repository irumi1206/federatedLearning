#!/usr/bin/env bash

# cluster based hierarchical federated learning is done on fixed enviroment of non-iid setting
# and with heterogeneous system setting. Dataset is cifar10, partitioned into 200 clients with dirichlet distribution of 0.1
# For the system heterogeneity, it is gruoped into 5 classes of devices each with computational time and communication time.
# To see how clustering clients affects the performance, following clustering is conducted.
# Because, the way of clustering is the key comparison matrix, sync-sync-aggregation will be used

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 200 \
        -dataheterogeneitytype dirichletdistribution \
        -systemheterogeneity realistic \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 10 \
        -clustersize 20 \
        -clustercommunicationtime 800 \
        -centralserverepoch 200 \
        -clusterepoch 2 \
        -localepoch 5

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 200 \
        -dataheterogeneitytype dirichletdistribution \
        -systemheterogeneity realistic \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 200 \
        -clustersize 1 \
        -clustercommunicationtime 800 \
        -centralserverepoch 200 \
        -clusterepoch 2 \
        -localepoch 5

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 200 \
        -dataheterogeneitytype dirichletdistribution \
        -systemheterogeneity realistic \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -clustersize 20 \
        -clustercommunicationtime 800 \
        -centralserverepoch 200 \
        -clusterepoch 2 \
        -localepoch 5

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 200 \
        -dataheterogeneitytype dirichletdistribution \
        -systemheterogeneity realistic \
        -clusteringtype clusterbygradientdisimilarity \
        -clusternum 10 \
        -clustersize 20 \
        -clustercommunicationtime 800 \
        -centralserverepoch 200 \
        -clusterepoch 2 \
        -localepoch 5