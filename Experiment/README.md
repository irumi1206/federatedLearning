Experiment 1

Experiment 1 is conducted to examine whether non-iid distribution amoun clients, or clusters affect the convergence more. For precise comparison, the training time for each cluster is set to be same via clustering. Estimated result is that cluster via gradient < random clustering < cluster via gradient alignment

- 95% onelabel dominant, 100clients, 10clients for each label
- For each label, computation time of {100,100,200,200,300,300,400,400,500,500}msec
- communication time with the edge server is x3 of the local computation time
- communication time with the cloud server is fixed to 800
- local epoch 5, group epoch 5
- clustered so that straggler(500) exist at least one per cluster
    - clustering it by random shuffle custom(to be runned)
    - clustering it by gradient similarity(done)
    - clustering it by gradient dissimilarity(running)