import torch.nn as nn
import torch
import numpy as np
import random
from utils import validate_model, get_model, get_optimizer

# Class for client, local device
class Client:
    def __init__(self, clusterid, clientid, dataloader, communicationtime, computationtimeperepoch, uniqueid,args):

        self.clusterid = clusterid
        self.clientid = clientid
        self.uniqueid = uniqueid
        self.dataloader = dataloader
        self.communicationtime = communicationtime
        self.computationtime = computationtimeperepoch
        self.model = get_model(args.modelname)
        self.model.to(args.device)
        self.optimizer = get_optimizer(self.model, args.optimizername, args.learningrate)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def calculate_training_time(self):

        # calculate the training time
        trainingtime = self.computationtime * self.args.localepoch + 2*self.communicationtime
        return trainingtime

    # logging is done by passing the queue due to the possibility of multi processing the clients in case of sync 
    def local_train(self,queue):

        seed = self.args.randomseed + self.uniqueid
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # reset optimizer
        self.optimizer = get_optimizer(self.model, self.args.optimizername, self.args.learningrate)

        # validate the model before training
        localaccuracybefore, _ = validate_model(self.model, self.dataloader, self.args)
        globalaccuracybefore, _= validate_model(self.model, self.args.testdataloader, self.args)

        # train the model
        for _ in range(self.args.localepoch):
            self.model.train()
            for x, y in self.dataloader:
                x,y = x.to(self.args.device), y.to(self.args.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
        
        # validate the model after training
    
        localaccuracyafter, _= validate_model(self.model, self.dataloader, self.args)
        globalaccuracyafter, _= validate_model(self.model, self.args.testdataloader, self.args)
        queue.put(f"{' '*94}<-> Client {self.clientid}, global : from {(100*globalaccuracybefore):.2f}% to {(100*globalaccuracyafter):.2f}%, local : from {(100*localaccuracybefore):.2f}% to {(100*localaccuracyafter):.2f}%, training time : {self.calculate_training_time()}msec")