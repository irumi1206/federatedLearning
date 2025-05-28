import torch
import torch.nn as nn
import logging
from utils import validate_model, get_model, validate_model_detailed
from queue import PriorityQueue
from torch.multiprocessing import Queue

class CentralServer:
    def __init__(self, args, clusterlist):

        self.clusterlist = clusterlist
        self.model = get_model(args.modelname)
        self.model.to(args.device)
        self.args = args

    def central_train(self):

        timepast = 0

        if self.args.interclusteringtype == "sync":

            for i in range(self.args.centralserverepoch):

                # validate the model before training
                logging.info(f"\n{'-'*100}\n")
                accuracybefore, lossbefore, accuracyperlabelbefore = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                logging.info(f"Central server round {i+1}, loss : {lossbefore:.2f}, accuracy :{(100*accuracybefore):.2f}%")
                logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")
                
                # load the model to the clusters
                for cluster in self.clusterlist:
                    cluster.model.load_state_dict(self.model.state_dict())
                    logging.info(f"{' '*42}->{cluster.communicationtime}msec")
                    cluster.cluster_train(i,i-1)
                    logging.info(f"{' '*42}{cluster.communicationtime}msec<-")

                # aggregate the models
                totalsamples = sum(
                sum(len(client.dataloader.dataset) for client in cluster.clientlist)
                for cluster in self.clusterlist
                )
                modelstatedict = self.model.state_dict()
                for key in modelstatedict:
                    modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                    for cluster in self.clusterlist:
                        clustersamples = sum(len(client.dataloader.dataset) for client in cluster.clientlist)
                        weight = clustersamples / totalsamples
                        modelstatedict[key] += cluster.model.state_dict()[key] * weight
                self.model.load_state_dict(modelstatedict)
                timepast += max([cluster.calculate_training_time() for cluster in self.clusterlist])

                # validate the model after training
                accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                logging.info(f"Central server round {i+1}, loss : {lossafter:.2f}, accuracy :{(100*accuracyafter):.2f}%")
                logging.info(f"Time past: {timepast}msec")
                logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")
                for cluster in self.clusterlist:
                    self.args.loggers[cluster.clusterid].info(f"Aggregated to central server at round {i+1} with accuracy {(100*accuracyafter):.2f}%")
                    self.args.loggers[cluster.clusterid].info(f"Loss : {lossafter}")
                    self.args.loggers[cluster.clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}\n")
                
                
        else:
            
            # priority queue to track the training sequence
            clustertimequeue = PriorityQueue()
            clustertoexecute = []
            for cluster in self.clusterlist:
                clustertrainingtime = cluster.calculate_training_time()
                clustertimequeue.put((clustertrainingtime, clustertrainingtime, cluster.clusterid))

            # calculate the client sequence
            timepast = 0
            for i in range(self.args.centralserverepoch * self.args.clusternum):
                timetoexecute, clustertime, clusterid = clustertimequeue.get()
                timepast = timetoexecute
                clustertimequeue.put((timetoexecute+clustertime,clustertime,clusterid))
                clustertoexecute.append((clusterid,timepast))

            # timestamp for each client and initialize each client's model
            # timstamp records the rouncd of which the models is loaded from central server after aggregation. For the very first aggregation, the timestamp is set to -1
            for cluster in self.clusterlist:
                cluster.model.load_state_dict(self.model.state_dict())
            timestamp = 0
            timestampforcluster = [-1] * len(self.clusterlist)

            
            # train the clients in the order of the sequence
            for epoch in range(self.args.centralserverepoch):
                for clusterind in range(self.args.clusternum):

                    # validate the model before training
                    logging.info(f"\n{'-'*100}\n")
                    accuracybefore, lossbefore, accuracyperlabelbefore = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                    logging.info(f"Central server round {timestamp+1}, loss : {lossbefore:.2f}, accuracy :{(100*accuracybefore):.2f}%")
                    logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")

                    # get cluster id and timepast to execute
                    clusterid, timepast = clustertoexecute[self.args.clusternum*epoch+clusterind]
                    
                    # local trainng the client
                    cluster = self.clusterlist[clusterid]
                    logging.info(f"{' '*42}->{cluster.communicationtime}msec")
                    cluster.cluster_train(timestamp,timestampforcluster[clusterid])

                    # calculate the staleness based function
                    alpha = self.args.interasyncalpha
                    staleness = timestamp - timestampforcluster[clusterid]
                    stalenessfunction = 1/(1+staleness)
                    alphatime = alpha * stalenessfunction
                    logging.info(f"{' '*42}Staleness : {staleness}")
                    logging.info(f"{' '*42}{cluster.communicationtime}msec<-")

                    # aggregate the cluster model and the client model
                    modelstatedict = self.model.state_dict()
                    for key in modelstatedict:
                        modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                        modelstatedict[key] = alphatime * cluster.model.state_dict()[key] + (1-alphatime) * self.model.state_dict()[key]
                    self.model.load_state_dict(modelstatedict)

                    # record the timestamp and load model parameter
                    timestampforcluster[clusterid] = timestamp
                    cluster.model.load_state_dict(self.model.state_dict())

                    # validate the model after training and log the accuracy log it to main logger, and to cluster logger as well
                    accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                    logging.info(f"Central server round {timestamp+1}, loss : {lossafter:.2f}, accuracy :{(100*accuracyafter):.2f}%")
                    logging.info(f"Time past : {timepast}msec")
                    logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")
                    self.args.loggers[clusterid].info(f"Aggregated to central server at round {timestamp+1} with staleness of {staleness}, aggregated accuracy of {(100*accuracyafter):.2f}%,")
                    self.args.loggers[clusterid].info(f"Loss : {lossafter}")
                    self.args.loggers[clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}\n")

                    # update the timestamp
                    timestamp += 1

