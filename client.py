import torch.nn as nn
import torch
import numpy as np
import random
from utils import validate_model, get_model, get_optimizer
from math import ceil
import copy

# Class for client, local device
class Client:
    def __init__(self, clusterid, clientid, dataloader, communicationtime, computationtime, uniqueid, localepoch, args):

        self.clusterid = clusterid
        self.clientid = clientid
        self.uniqueid = uniqueid
        self.dataloader = dataloader
        self.communicationtime = communicationtime
        self.computationtime = computationtime
        self.model = get_model(args.modelname)
        self.optimizer = get_optimizer(self.model, args.optimizername, args.learningrate)
        self.criterion = nn.CrossEntropyLoss()
        self.localepoch = localepoch
        self.args = args

    def calculate_training_time(self):
        # calculate the training time
        if self.args.computationcapabilitymatric == "bybatch": trainingtime = self.computationtime * self.localepoch * ceil(len(self.dataloader.dataset) / int(self.dataloader.batch_size)) + 2*self.communicationtime
        elif self.args.computationcapabilitymatric == "byepoch": trainingtime = self.computationtime * self.localepoch + + 2*self.communicationtime
        elif self.args.computationcapabilitymatric == "fixed": trainingtime = self.computationtime + 2*self.communicationtime
        else: raise ValueError("not implemented")
        return trainingtime

    # logging is done by passing the queue due to the possibility of multi processing the clients in case of sync 
    def local_train(self,queue):

        # reset optimizer
        self.optimizer = get_optimizer(self.model, self.args.optimizername, self.args.learningrate)

        # validate the model before training
        localaccuracybefore, locallossbefore = validate_model(self.model, self.dataloader, self.args)
        globalaccuracybefore, globallossbefore= validate_model(self.model, self.args.testdataloader, self.args)

        modelbefore = copy.deepcopy(self.model)
        # train the model
        for _ in range(self.localepoch):
            self.model.train()
            for x, y in self.dataloader:
                x,y = x.to(self.args.device), y.to(self.args.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                if self.args.regularizationcoefficient !=  0.0:
                    proxterm = 0.0
                    for localweight, globalweight in zip(self.model.parameters(), modelbefore.parameters()):
                        proxterm += ((localweight - globalweight.detach())**2).sum()
                    loss += (self.args.regularizationcoefficient/2)*proxterm

                loss.backward()
                self.optimizer.step()
        
        # validate the model after training
        localaccuracyafter, locallossafter= validate_model(self.model, self.dataloader, self.args)
        globalaccuracyafter, globallossafter= validate_model(self.model, self.args.testdataloader, self.args)
        queue.put(f"{' '*94}<-> Client {self.clientid}, global : from {(100*globalaccuracybefore):.2f}% to {(100*globalaccuracyafter):.2f}%, local : from {(100*localaccuracybefore):.2f}% to {(100*localaccuracyafter):.2f}%, globalloss : from {globallossbefore:.3f} to {globallossafter:.3f}, localloss : from {locallossbefore:.3f} to {locallossafter:.3f}, training time : {self.calculate_training_time()}msec")

        # calculate the training time
        trainingtime = self.calculate_training_time()
        datasize = len(self.dataloader.dataset)
        return trainingtime, datasize
