#!/usr/bin/env bash

# bash script for lab machine 2

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.1 \
        -learningratedecay 0.992

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilarity \
        -clusternum 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.1 \
        -learningratedecay 0.992

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientbetweenassumeonelabeldominant100clients \
        -clusternum 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.1 \
        -learningratedecay 0.992

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.1 \
        -learningratedecay 0.992