#!/usr/bin/env bash

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
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

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
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
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity custom2 \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdisimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
       -clusterepochtype fixed \
        -clusterepoch 5 \
        -localepochtype fixed \
        -localepoch 5