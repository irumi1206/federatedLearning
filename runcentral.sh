#!/usr/bin/env bash

# clusterbyclientorder
# clusterbyrandomshuffle
# dirichletdistribution
# iid

# expeirment 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype iid \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 3


# expeirment 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype fixed \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.1 \
        -clusterparticipationratio 10 \
        -randomseed 3

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 1

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 2

python3 main.py \
        -interclusteringtype async \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onelabeldominant \
        -systemheterogeneity custom \
        -clusteringtype clusterbyclientorder \
        -clusternum 100 \
        -clustersize 1 \
        -clustercommunicationtime 0 \
        -centralserverepoch 100 \
        -clusterepochtype fixed \
        -clusterepoch 1 \
        -localepochtype custom2 \
        -localepoch 5 \
        -regularizationcoefficient 0.01 \
        -clusterparticipationratio 10 \
        -randomseed 3


