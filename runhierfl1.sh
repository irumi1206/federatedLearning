#!/usr/bin/env bash

# python3 main.py \
#         -interclusteringtype sync \
#         -intraclusteringtype sync \
#         -modelname nnmnist \
#         -datasetname mnist \
#         -clientnum 10 \
#         -dataheterogeneitytype onelabeldominant \
#         -computationcapabilitymatric fixed\
#         -systemheterogeneity alltimesame \
#         -clustercommunicationtime 800 \
#         -clusteringtype clusterbyrandomshuffle \
#         -clusternum 5 \
#         -clustersize 2 \
#         -centralserverepoch 100 \
#         -clusterepochtype fixed \
#         -clusterepoch 2 \
#         -localepochtype fixed \
#         -localepoch 3

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 10 \
        -dataheterogeneitytype iid \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 5 \
        -clustersize 2 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 2 \
        -localepochtype fixed \
        -localepoch 3

# python3 main.py \
#         -interclusteringtype sync \
#         -intraclusteringtype sync \
#         -modelname cnnfemnist \
#         -datasetname femnist \
#         -clientnum 100 \
#         -dataheterogeneitytype onelabeldominant \
#         -computationcapabilitymatric fixed\
#         -systemheterogeneity alltimesame \
#         -clustercommunicationtime 800 \
#         -clusteringtype clusterbyrandomshuffle \
#         -clusternum 10\
#         -clustersize 10 \
#         -centralserverepoch 100 \
#         -clusterepochtype fixed \
#         -clusterepoch 2 \
#         -localepochtype fixed \
#         -localepoch 3



