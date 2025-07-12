#!/usr/bin/env bash

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype dirichletdistribution \
        -computationcapabilitymatric fixed\
        -systemheterogeneity realistic \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype dirichletdistribution \
        -computationcapabilitymatric fixed\
        -systemheterogeneity realistic \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype dirichletdistribution \
        -computationcapabilitymatric fixed\
        -systemheterogeneity realistic \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilarityandsystemsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype dirichletdistribution \
        -computationcapabilitymatric fixed\
        -systemheterogeneity realistic \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2



