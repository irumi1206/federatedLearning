#!/usr/bin/env bash

# upon random system capability assignment, the training time for each cluster is not same, leading to fast cluster
# hindering the convergence performance

# experiment 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity custom2 \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 5 \
        -localepochtype fixed \
        -localepoch 5

# experiment 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity custom2 \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 5 \
        -localepochtype fixed \
        -localepoch 5

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity custom2 \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 5 \
        -localepochtype fixed \
        -localepoch 5
