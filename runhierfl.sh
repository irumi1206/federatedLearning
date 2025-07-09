#!/usr/bin/env bash

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity assume100clients95onelabeldominance_foreachlabel100200300400500of2randomlydistributed_comm3 \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdisimilarityandsystemsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 5 \
        -localepochtype fixed \
        -localepoch 5


