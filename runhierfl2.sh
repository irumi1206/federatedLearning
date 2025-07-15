#!/usr/bin/env bash

# bash script for machine2 at the lab

# python3 main.py \
#         -interclusteringtype async \
#         -intraclusteringtype sync \
#         -modelname cnncifar10 \
#         -datasetname cifar10 \
#         -clientnum 100 \
#         -dataheterogeneitytype onelabeldominant \
#         -computationcapabilitymatric fixed\
#         -systemheterogeneity assume100clients95onelabeldominance_foreachlabel100200300400500of2randomlydistributed_comm3 \
#         -clustercommunicationtime 800 \
#         -clusteringtype clusterbygradientsimilarity \
#         -clusternum 10 \
#         -clustersize 10 \
#         -centralserverepoch 100 \
#         -clusterepochtype fixed \
#         -clusterepoch 10 \
#         -localepochtype fixed \
#         -localepoch 2

# python3 main.py \
#         -interclusteringtype async \
#         -intraclusteringtype sync \
#         -modelname cnncifar10 \
#         -datasetname cifar10 \
#         -clientnum 100 \
#         -dataheterogeneitytype onelabeldominant \
#         -computationcapabilitymatric fixed\
#         -systemheterogeneity assume100clients95onelabeldominance_foreachlabel100200300400500of2randomlydistributed_comm3 \
#         -clustercommunicationtime 800 \
#         -clusteringtype clusterbygradientdissimilarity \
#         -clusternum 10 \
#         -clustersize 10 \
#         -centralserverepoch 100 \
#         -clusterepochtype fixed \
#         -clusterepoch 10 \
#         -localepochtype fixed \
#         -localepoch 2

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
        -clusteringtype clusterbygradientdissimilarityandsystemsimilarity \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0

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
        -clusteringtype clusterbyrandomshufflecustom \
        -clusternum 10 \
        -clustersize 10 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0



