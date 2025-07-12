#!/usr/bin/env bash

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname nnmnist \
        -datasetname mnist \
        -clientnum 10 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 5 \
        -clustersize 2 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 2 \
        -localepochtype fixed \
        -localepoch 3