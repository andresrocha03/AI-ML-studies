import os
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import utils
import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset
import utils
import argparse


data_folder = '/home/andre/unicamp/ini_cien/machine_learning/federated_learning/practicing/data/processed'
x_treino = pd.read_csv(os.path.join(data_folder, 'x_treino.csv'))
y_treino = pd.read_csv(os.path.join(data_folder, 'y_treino.csv'))
x_teste = pd.read_csv(os.path.join(data_folder, 'x_teste.csv'))
y_teste = pd.read_csv(os.path.join(data_folder, 'y_teste.csv'))

df_treino = x_treino.copy()
df_treino['label'] = y_treino
df_teste = x_teste.copy()
df_teste['label'] = y_teste

class SimpleClient(fl.client.NumPyClient):
    def __init__(self, x_treino, y_treino, x_teste, y_teste):
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.x_teste = x_teste
        self.y_teste = y_teste
        self.model = LogisticRegression(warm_start=True, solver="saga", max_iter=1)
        utils.set_initial_params(self.model)

    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)
    def set_parameters(self, parameters):
        return utils.set_model_params(self.model, parameters)

    def fit(self,parameters,config):     
        self.model = self.set_parameters(parameters)
        self.model.fit(self.x_treino, self.y_treino)
        return utils.get_model_parameters(self.model), len(self.x_treino), {}


    def evaluate(self,parameters,config):
        self.model = self.set_parameters(parameters)
        y_prob = self.model.predict_proba(self.x_teste)
        scores = utils.get_scores(self.y_teste, y_prob)
        print(f"Client loss, accuracy, precision and recall for {scores['loss']}, {scores['accuracy']}, {scores['precision']}, {scores['recall']}")
        return scores["loss"], len(self.x_teste), scores





def create_client(cid: str):
    #get train and test data
    X_train, y_train = utils.load_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_data(test_partitions[int(cid)-1])
    return SimpleClient(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #client id
    parser.add_argument("--id", type=str, help="Client id")
    args = parser.parse_args()

    # partition the data for 3 cleints
    num_clients = 3
    train_partitions = utils.partition_data(df_treino, num_clients)
    test_partitions = utils.partition_data(df_teste, num_clients)

    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(args.id).to_client())    
