python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
<<<<<<< HEAD
        -dataheterogeneitytype onlyspecificlabel \
=======
        -dataheterogeneitytype onelabeldominant \
        -dirichletalpha 0.01 \
>>>>>>> 791c875 (cluster)
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilaritygreedy \
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
<<<<<<< HEAD
        -clusterfrequency 200
=======
        -clusterfrequency 200 \
        -balancelambda 0.0 \
        -diversitylambda
>>>>>>> 791c875 (cluster)

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
<<<<<<< HEAD
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilarity \
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
        -clusterfrequency 200

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbyrandomshuffle \
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
        -clusterfrequency 200

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientbetween \
=======
        -dataheterogeneitytype dirichletdistribution \
        -dirichletalpha 0.01 \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientdissimilaritygreedy \
>>>>>>> 791c875 (cluster)
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
        -clusterfrequency 200 \
<<<<<<< HEAD
        -roundrobinlevel 2

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientbetween \
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
        -clusterfrequency 200 \
        -roundrobinlevel 5

python3 main.py \
        -interclusteringtype sync \
        -intraclusteringtype sync \
        -modelname cnncifar10 \
        -datasetname cifar10 \
        -clientnum 100 \
        -dataheterogeneitytype onlyspecificlabel \
        -computationcapabilitymatric fixed\
        -systemheterogeneity alltimesame \
        -clustercommunicationtime 800 \
        -clusteringtype clusterbygradientsimilarity \
        -clusternum 10 \
        -centralserverepoch 200 \
        -clusterepochtype fixed \
        -clusterepoch 10 \
        -localepochtype fixed \
        -localepoch 2 \
        -examinethemodelindetail 0 \
        -learningrate 0.01 \
        -clusterfrequency 200
=======
        -balancelambda 0.0 \
        -diversitylambda 0.5
>>>>>>> 791c875 (cluster)
