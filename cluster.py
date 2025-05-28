import torch
import torch.nn as nn
import logging
from utils import validate_model, get_model, validate_model_detailed
from torch.multiprocessing import Process, Queue
from queue import PriorityQueue
def train_single_client(client,queue):
    client.local_train(queue)

class Cluster:
    def __init__(self, clusterid, communicationtime, args, clientlist):

        self.clusterid = clusterid
        self.communicationtime = communicationtime
        self.clientlist = clientlist
        self.model = get_model(args.modelname)
        self.model.to(args.device)
        self.testdataloader = args.testdataloader
        self.intraclusteringtype = args.intraclusteringtype
        self.device = args.device
        self.clusterepoch = args.clusterepoch
        self.clustersize = args.clustersize
        self.args = args

    def calculate_training_time(self):

        # calculate the training time
        if self.intraclusteringtype == "sync":

            # maximum training time amoung all clients
            trainingtime = 0
            for client in self.clientlist:
                trainingtime = max(trainingtime, client.calculate_training_time())
            return trainingtime * self.clusterepoch + 2 *self.communicationtime
        
        elif self.intraclusteringtype == "async":

            # priority queue for the training time of each client
            clienttimequeue = PriorityQueue()
            for client in self.clientlist:
                clienttimequeue.put((client.calculate_training_time(), client.calculate_training_time(), client.clientid))

            # time past
            timepast = 0
            for i in range(self.clusterepoch * self.clustersize):
                timetoexecute, clienttime, clientid = clienttimequeue.get()
                timepast = timetoexecute
                clienttimequeue.put((timetoexecute+clienttime,clienttime,clientid))
                

            return timepast + 2 * self.communicationtime

    def cluster_train(self, roundnumber, modelroundnumber):

        # assuming that it got model parameters from the central server, communication time past
        timepast = self.communicationtime

        if self.intraclusteringtype == "sync":

            # validate the model before training
            accuracybefore, lossbefore, accuracyperlabelbefore = validate_model_detailed(self.model, self.testdataloader, self.args)
            self.args.loggers[self.clusterid].info(f"{'-'*53}\n")
            self.args.loggers[self.clusterid].info(f"Central server round {roundnumber+1} started")
            self.args.loggers[self.clusterid].info(f"Model sent from central server with {timepast}msec communication time, with loss :{lossbefore:.2f}, accuracy : {(100*accuracybefore):.2f}%")
            self.args.loggers[self.clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")
            self.args.loggers[self.clusterid].info(f"The model is from round {modelroundnumber+1} from central server, staleness is {roundnumber-modelroundnumber-1}\n")
            logging.info(f"{' '*53}-> Cluster {self.clusterid}, loss : {lossbefore:.2f}, accuracy {(100*accuracybefore):.2f}%, model from round {modelroundnumber+1}")
            logging.info(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")

            for i in range(self.clusterepoch):
                
                # load the model from the central cluster
                for client in self.clientlist:
                    client.model.load_state_dict(self.model.state_dict())
                    
                # use multi process, it does not use gpu simultaneously, other operations are in parallel
                processes = []
                queue = Queue()
                for client in self.clientlist:
                    p = Process(target=train_single_client, args=(client,queue))
                    p.start()
                    processes.append(p)

                # Wait for all to finish
                for p in processes:
                    p.join()
                
                #log the results
                while not queue.empty():
                    logging.info(queue.get())

                # aggregate the models based on the dataset size
                totalsamples = sum(len(client.dataloader.dataset) for client in self.clientlist)
                modelstatedict = self.model.state_dict()
                for key in modelstatedict:
                    modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                    for client in self.clientlist:
                        weight = len(client.dataloader.dataset) / totalsamples
                        modelstatedict[key] += client.model.state_dict()[key] * weight
                self.model.load_state_dict(modelstatedict)

                # after training, keep track of timepast
                timeperepoch = max(client.calculate_training_time() for client in self.clientlist)
                timepast += timeperepoch

                # validate the model after training and log it to overall training log
                accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.testdataloader, self.args)
                logging.info(f"{' '*53}Cluster {self.clusterid}, round {i+1}, loss : {lossafter:.2f}, accuracy {(100*accuracyafter):.2f}% <-")
                logging.info(f"{' '*53}Time past : {timepast}msec")
                logging.info(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")

                # log the information to cluster level logger
                self.args.loggers[self.clusterid].info(f"Cluster round {i+1} completed, aggregated models from all the clients, time past : {timepast}msec, loss : {lossafter:.2f}, accuracy {(100*accuracyafter):.2f}%")
                self.args.loggers[self.clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}\n")

        else:

            # priority queue to track the training sequence
            clienttimequeue = PriorityQueue()
            clienttoexecute = []
            for client in self.clientlist:
                clienttrainingtime = client.calculate_training_time()
                clienttimequeue.put((clienttrainingtime+timepast, clienttrainingtime, client.clientid))

            # calculate the client sequence
            for i in range(self.clusterepoch * self.clustersize):
                timetoexecute, clienttime, clientid = clienttimequeue.get()
                timepast = timetoexecute
                clienttimequeue.put((timetoexecute+clienttime,clienttime,clientid))
                clienttoexecute.append((clientid,timepast))

            # timestamp for each client and initialize each client's model
            for client in self.clientlist:
                client.model.load_state_dict(self.model.state_dict())
            timestamp = 0 
            timestampforclient = [timestamp] * len(self.clientlist)
            
            # validate the model before training
            accuracybefore, lossbefore, accuracyperlabelbefore = validate_model_detailed(self.model, self.testdataloader, self.args)
            self.args.loggers[self.clusterid].info(f"{'-'*53}\n")
            self.args.loggers[self.clusterid].info(f"Central server round {roundnumber+1} started")
            self.args.loggers[self.clusterid].info(f"Model sent from central server with {timepast}msec communication time, with loss :{lossbefore:.2f}, accuracy : {(100*accuracybefore):.2f}%")
            self.args.loggers[self.clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")
            self.args.loggers[self.clusterid].info(f"The model is from round {modelroundnumber+1} from central server, staleness is {roundnumber-modelroundnumber-1}\n")
            logging.info(f"{' '*53}-> Cluster {self.clusterid}, loss : {lossbefore:.2f}, accuracy {(100*accuracybefore):.2f}%, model from round {modelroundnumber+1}")
            logging.info(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelbefore.items()]}")


            # train the clients in the order of the sequence
            for epoch in range(self.clusterepoch):    
                for clientind in range(self.clustersize):
                    
                    # get client id and timepast to execute
                    clientid, timepast = clienttoexecute[self.clustersize*epoch+clientind]
                    
                    # local trainng the client
                    client = self.clientlist[clientid]
                    queue = Queue()
                    client.local_train(queue)

                    # calulate the staleness and log device
                    alpha = self.args.intraasyncalpha
                    staleness = timestamp - timestampforclient[clientid]
                    stalenessfunction = 1/(1+staleness)
                    alphatime = alpha * stalenessfunction
                    while not queue.empty():
                        logging.info(queue.get()+f", staleness :{staleness}")

                    # aggregate the cluster model and the client model
                    modelstatedict = self.model.state_dict()
                    for key in modelstatedict:
                        modelstatedict[key] = torch.zeros_like(modelstatedict[key])
                        modelstatedict[key] = alphatime * client.model.state_dict()[key] + (1-alphatime) * self.model.state_dict()[key]
                    self.model.load_state_dict(modelstatedict)

                    # validate the model after training and log it to overall training log
                    accuracyafter, lossafter, accuracyperlabelafter = validate_model_detailed(self.model, self.testdataloader, self.args)
                    logging.info(f"{' '*53}Cluster {self.clusterid}, round {timestamp+1}, accuracy {(100*accuracyafter):.2f}% <-")
                    logging.info(f"{' '*53}Time past : {timepast}msec")
                    logging.info(f"{' '*53}{[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}")

                    # log the information to cluster level logger
                    self.args.loggers[self.clusterid].info(f"Cluster round {timestamp+1} completed, aggregated model from client {clientid}, time past : {timepast}msec, loss : {lossafter:.2f}, accuracy {(100*accuracyafter):.2f}%")
                    self.args.loggers[self.clusterid].info(f"Accuracy per label : {[f'{label}:{(accuracy*100):.2f}%' for label, accuracy in accuracyperlabelafter.items()]}\n")
                    
                    # update the timestamp
                    timestamp += 1

                    # schedule the next client. currently, the client trained before is trained right afterward
                    client.model.load_state_dict(self.model.state_dict())
                    timestampforclient[clientid] = timestamp         