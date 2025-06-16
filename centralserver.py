import torch
import torch.nn as nn
import logging
from utils import get_model, validate_model_detailed
from queue import PriorityQueue
import random
from queue import Queue
from collections import defaultdict

class CentralServer:
    def __init__(self, interclusteringtype, centralserverepoch, args, clusterlist):

        self.interclusteringtype = interclusteringtype
        self.centralserverepoch = centralserverepoch
        self.clusterlist = clusterlist
        self.model = get_model(args.modelname)
        self.model.to(args.device)
        self.args = args

    def central_train(self):

        timepast = 0
        datasize = 0

        # validate the model before training and log it to overall training log file
        logging.info(f"\n{'-'*100}\n")
        accuracyinitial, lossinitial, accuracyperlabelinitial = validate_model_detailed(self.model, self.args.testdataloader, self.args)
        logging.info(f"Central server round 1, loss : {lossinitial:.2f}, accuracy :{(100*accuracyinitial):.2f}%")
        logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelinitial.items()]}")
        self.args.centralservertimepast.append(timepast)
        self.args.centralserverround.append(0)
        self.args.centralserveraccuracy.append(accuracyinitial)
        self.args.centralserverloss.append(lossinitial)

        if self.interclusteringtype == "sync":

            for i in range(self.centralserverepoch):
                
                clusternum = len(self.clusterlist)
                participatingclusternum = max(1,int(clusternum * self.args.clusterparticipationratio/100))
                participatingclusterind = random.sample([i for i in range(clusternum)], participatingclusternum)            

                # load the model to the clusters
                epochtimepast = 0
                epochdatasize = 0
                datasizecached = defaultdict(int)
                for clusterind in participatingclusterind:
                    cluster= self.clusterlist[clusterind]
                    cluster.model.load_state_dict(self.model.state_dict())
                    logging.info(f"{' '*42}->{cluster.communicationtime}msec")
                    messagequeue = Queue()
                    t, d = cluster.cluster_train(i,messagequeue)
                    while not messagequeue.empty():
                        logging.info(messagequeue.get())
                    epochtimepast = max(epochtimepast,t)
                    epochdatasize += d
                    datasizecached[clusterind] = d
                    logging.info(f"{' '*42}{cluster.communicationtime}msec<-")
                timepast +=epochtimepast
                datasize += epochdatasize

                # aggregate the models
                modelstatedict = self.model.state_dict()
                for key in modelstatedict:
                    modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                    for clusterind in participatingclusterind:
                        cluster=self.clusterlist[clusterind]
                        clustersamples = datasizecached[clusterind]
                        weight = clustersamples / epochdatasize
                        modelstatedict[key] += cluster.model.state_dict()[key] * weight
                self.model.load_state_dict(modelstatedict)
                datasizecached.clear()

                # validate the model after training and log it to overall training log file
                accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                logging.info(f"Central server round {i+1}, loss : {lossafter:.2f}, accuracy :{(100*accuracyafter):.2f}%")
                logging.info(f"Time past: {timepast}msec")
                logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")
                self.args.centralservertimepast.append(timepast)
                self.args.centralserverround.append(i+1)
                self.args.centralserveraccuracy.append(accuracyafter)
                self.args.centralserverloss.append(lossafter)

                     
        else:

            clusternum = len(self.clusterlist)
            participatingclusternum = max(1,int(clusternum * self.args.clusterparticipationratio/100))
            participatingclusterind = random.sample([i for i in range(clusternum)], participatingclusternum)
            notparticipatingclusterind = [i for i in range(clusternum) if i not in participatingclusterind ]

            # keep track of timestamp to calculate staleness
            modelversionforcluster = [-1] * len(self.clusterlist)
            modelversion = 0
            loggingcached = defaultdict(list)

            clustereventqueue = PriorityQueue()
            for ind in participatingclusterind:
                cluster = self.clusterlist[ind]
                #logging.info(f"{' '*42}->{cluster.communicationtime}msec")
                loggingcached[ind].append(f"{' '*42}->{cluster.communicationtime}msec")
                cluster.model.load_state_dict(self.model.state_dict())
                modelversionforcluster[ind] = modelversion
                messagequeue = Queue()
                t, d = cluster.cluster_train(0, messagequeue)
                while not messagequeue.empty():
                    loggingcached[ind].append(messagequeue.get())
                clustereventqueue.put((t,ind))

            round = 0

            for epoch in range(self.centralserverepoch):
                for _ in range(participatingclusternum):
                    # get the cluster that fast arrived
                    arrivedtime, arrivedclusterind = clustereventqueue.get()
                    timepast = arrivedtime
                    cluster = self.clusterlist[arrivedclusterind]
                    loggingmessage = loggingcached[arrivedclusterind]
                    for message in loggingmessage:
                        logging.info(message)

                    # aggregate it
                    
                    alpha = self.args.interasyncalpha
                    staleness = modelversion - modelversionforcluster[arrivedclusterind]

                    if staleness <=self.args.interasyncthreshold:
                        stalenessfunction = 1/(1+staleness)
                        alphatime = alpha * stalenessfunction
                        logging.info(f"{' '*42}Staleness : {staleness}")
                        logging.info(f"{' '*42}{cluster.communicationtime}msec<-")
                        modelstatedict = self.model.state_dict()
                        for key in modelstatedict:
                            #modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                            modelstatedict[key] = alphatime * cluster.model.state_dict()[key] + (1-alphatime) * self.model.state_dict()[key]
                        self.model.load_state_dict(modelstatedict)
                        modelversion +=1

                    # pick one client
                    participatingclusterind.remove(arrivedclusterind)
                    notparticipatingclusterind.append(arrivedclusterind)
                    pickedclusterind = random.sample(notparticipatingclusterind, 1)[0]
                    notparticipatingclusterind.remove(pickedclusterind)
                    participatingclusterind.append(pickedclusterind)
                    self.clusterlist[pickedclusterind].model.load_state_dict(self.model.state_dict())
                    modelversionforcluster[pickedclusterind] = modelversion
                    messagequeue = Queue()
                    t,d = self.clusterlist[pickedclusterind].cluster_train(modelversion,messagequeue)
                    loggingcached[pickedclusterind].clear()
                    loggingcached[pickedclusterind].append(f"{' '*42}->{self.clusterlist[pickedclusterind].communicationtime}msec")
                    while not messagequeue.empty():
                        loggingcached[pickedclusterind].append(messagequeue.get())
                    clustereventqueue.put((timepast+t, pickedclusterind))

                    # validate the model after training and log the accuracy log it to main logger, and to cluster logger as well
                    accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.args.testdataloader, self.args)
                    logging.info(f"Central server round {round+1}, loss : {lossafter:.2f}, accuracy :{(100*accuracyafter):.2f}%")
                    logging.info(f"Time past : {timepast}msec")
                    logging.info(f"{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")
                    
                    round +=1

                self.args.centralservertimepast.append(timepast)
                self.args.centralserverround.append(epoch+1)
                self.args.centralserveraccuracy.append(accuracyafter)
                self.args.centralserverloss.append(lossafter)   


