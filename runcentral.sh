#!/usr/bin/env bash

# overall setting : 100 clients with iid setting from cifar 10 dataset, 10% participation, iid and with system heterogeneity
# 1. sync, local epoch 3
# 2. sync, local epoch 5
# 3. async with alpha base 0.5, threshold 5, local epoch 3
# 3. async with alpha base 0.5, threshold 5, local epoch 5
# 4. async with alpha base 0.5, threshold 10, local epoch 3
# 3. async with alpha base 0.5, threshold 10, local epoch 5


python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 3 \
        -clusterparticipationratio 10

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 5 \
        -clusterparticipationratio 10

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 3 \
        -interasyncalpha 0.5 \
        -interasyncthreshold 5 \
        -clusterparticipationratio 10

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 5 \
        -interasyncalpha 0.5 \
        -interasyncthreshold 5 \
        -clusterparticipationratio 10

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 3 \
        -interasyncalpha 0.5 \
        -interasyncthreshold 10 \
        -clusterparticipationratio 10
                
python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity communicationtimesamecomputationdifferent \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepoch 1 \
        -localepoch 5 \
        -interasyncalpha 0.5 \
        -interasyncthreshold 10 \
        -clusterparticipationratio 10
