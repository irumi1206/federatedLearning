#!/usr/bin/env bash

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
        -clusternum 100 \
        -clustersize 1 \
        -centralserverepoch 2 \
        -clusterepoch 2 \
        -localepoch 2 \
        -clusterparticipationratio 10
        