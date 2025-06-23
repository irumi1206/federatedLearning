import torch
import torch.nn as nn
from utils import get_model, validate_model_detailed
from queue import PriorityQueue
from queue import Queue
import random
from collections import defaultdict

class Cluster:
    def __init__(self, clusterid, communicationtime, intraclusteringtype, clusterepoch, args, clientlist):

        self.clusterid = clusterid
        self.communicationtime = communicationtime
        self.intraclusteringtype = intraclusteringtype
        self.clusterepoch = clusterepoch
        self.args = args
        self.model = get_model(args.modelname)
        self.model.to(args.device)
        self.testdataloader = args.testdataloader
        self.clientlist = clientlist


    def cluster_train(self,modelroundnumber,messagequeue):

        # assuming that it got model parameters from the central server, communication time past
        timepast = self.communicationtime
        datasize = 0

        # validate the model before training
        accuracybefore, lossbefore, accuracyperlabelbefore = validate_model_detailed(self.model, self.testdataloader, self.args)
        messagequeue.put(f"{' '*53}-> Cluster {self.clusterid}, loss : {lossbefore:.2f}, accuracy {(100*accuracybefore):.2f}%, model from round {modelroundnumber+1}")
        messagequeue.put(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")

        if self.intraclusteringtype == "sync":

            for i in range(self.clusterepoch):

                clientnum = len(self.clientlist)
                selectedclientnum = max(1,int(clientnum*self.args.clientparticipationratio/100))
                selectedclientind = random.sample([i for i in range(clientnum)], selectedclientnum)
                
                # load the model from the central cluster
                epochtimepast = 0
                epochdatasize = 0
                datasizecached = defaultdict(int)
                q = Queue()
                for clientind in selectedclientind:
                    client=self.clientlist[clientind]
                    client.model.to(self.args.device)
                    client.model.load_state_dict(self.model.state_dict())
                    t, d = client.local_train(q)
                    epochtimepast = max(epochtimepast, t)
                    epochdatasize += d
                    datasizecached[clientind] = d
                datasize += epochdatasize
                timepast += epochtimepast
                
                
                #log the results
                while not q.empty():
                    messagequeue.put(q.get())

                # aggregate the models based on the dataset size
                modelstatedict = self.model.state_dict()
                for key in modelstatedict:
                    modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                    for clientind in selectedclientind:
                        client = self.clientlist[clientind]
                        weight = datasizecached[clientind] / epochdatasize
                        modelstatedict[key] += client.model.state_dict()[key] * weight
                self.model.load_state_dict(modelstatedict)
                datasizecached.clear()

                for clientind in selectedclientind:
                    client=self.clientlist[clientind]
                    client.model.to("cpu")

                # validate the model after training and log it to overall training log
                accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.testdataloader, self.args)
                messagequeue.put(f"{' '*53}Cluster {self.clusterid}, round {i+1}, loss : {lossafter:.2f}, accuracy {(100*accuracyafter):.2f}% <-")
                messagequeue.put(f"{' '*53}Time past : {timepast}msec")
                messagequeue.put(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")

            timepast += self.communicationtime

            return timepast, datasize
        
        else:
            
            # pick random clients
            clientnum = len(self.clientlist)
            participatingclientnum = max(1,int(clientnum * self.args.clientparticipationratio/100))
            participatingclientind = random.sample([i for i in range(clientnum)], participatingclientnum)
            notparticipatingclientind = [i for i in range(clientnum) if i not in participatingclientind ]

            # keep track of timestamp to calculate staleness
            clientmodelversion = [-1] * len(self.clientlist)
            modelversion = 0

            # generate eventqueue and train some clients
            clienteventqueue = PriorityQueue()
            datasizecached = defaultdict(int)
            loggingcached = defaultdict(list)
            for ind in participatingclientind:
                queue = Queue()
                client = self.clientlist[ind]
                client.model.load_state_dict(self.model.state_dict())
                clientmodelversion[ind] = modelversion
                t, d = client.local_train(queue)
                datasizecached[ind] = d
                while not queue.empty():
                        loggingcached[ind].append(queue.get())
                clienteventqueue.put((t+timepast,ind))

            round = 0

            # pop the event with least executiontime, and pick anotherone for the event
            for epoch in range(self.clusterepoch):
                for _ in range(participatingclientnum):

                    # get the client that fast arrived
                    arrivedtime, arrivedclientind = clienteventqueue.get()
                    timepast = arrivedtime
                    client = self.clientlist[arrivedclientind]
                    loggingmessage = loggingcached[arrivedclientind]
                    for message in loggingmessage:
                        messagequeue.put(message)

                    # aggregate it
                    alpha = self.args.intraasyncalpha
                    staleness = modelversion - clientmodelversion[arrivedclientind]
                    if staleness<=self.args.intraasyncthreshold:
                        stalenessfunction = 1/(1+staleness)
                        alphatime = alpha * stalenessfunction
                        modelstatedict = self.model.state_dict()
                        for key in modelstatedict:
                            #modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                            modelstatedict[key] = alphatime * client.model.state_dict()[key] + (1-alphatime) * self.model.state_dict()[key]
                        self.model.load_state_dict(modelstatedict)
                        modelversion +=1

                    # pick one client
                    participatingclientind.remove(arrivedclientind)
                    notparticipatingclientind.append(arrivedclientind)
                    pickedclientind = random.sample(notparticipatingclientind, 1)[0]
                    notparticipatingclientind.remove(pickedclientind)
                    participatingclientind.append(pickedclientind)
                    
                    self.clientlist[pickedclientind].model.load_state_dict(self.model.state_dict())
                    clientmodelversion[pickedclientind] = modelversion
                    q = Queue()
                    t,d = self.clientlist[pickedclientind].local_train(q)
                    datasizecached[pickedclientind] = d
                    loggingcached[pickedclientind].clear()
                    while not q.empty():
                        loggingcached[pickedclientind].append(q.get())
                    clienteventqueue.put((timepast+t, pickedclientind))

                    # validate the model after training and log it to overall training log
                    accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.testdataloader, self.args)
                    messagequeue.put(f"{' '*53}Cluster {self.clusterid}, round {round+1}, accuracy {(100*accuracyafter):.2f}% <-")
                    messagequeue.put(f"{' '*53}Time past : {timepast}msec")
                    messagequeue.put(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")

                    round +=1

            datasize += sum(datasizecached.values())

            timepast += self.communicationtime

            return timepast, datasize
         