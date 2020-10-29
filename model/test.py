import pickle
import yaml
import os
import sys
import numpy as np
import math
import time
import torch
from model import SDPP

if __name__ == "__main__":

    config_path = "config/main.yaml"
    file_config = open(config_path,'r')
    file_data = file_config.read()
    yaml_config = yaml.load(file_data)
    preprocess_config = yaml_config["preprocess"]

    n_nodes,n_sequences,n_steps = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["information"]),'rb'))
    x_train, y_train, sz_train,time_train,rnn_index_train, vocabulary_size = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["train"]),'rb'))
    x_test, y_test, sz_test,time_test,rnn_index_test, _ = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["test"]),'rb'))
    x_val, y_val, sz_val, time_val,rnn_index_val,_ = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["val"]),'rb'))


    config = {
        "n_sequences": n_sequences,
        "n_steps": n_steps,
        "time_interval": preprocess_config["time-interval"],
        "n_time_interval": preprocess_config["n-time-interval"],
        "learning_rate": 0.1,
        "sequence_batch_size": 20,
        "batch_size": 32,
        "n_hidden_gru": 32,
        "l1": 5e-5,
        "l2": 0.1,
        "l1l2": 1.0,
        "activation": "relu",
        "training_iters": 200*3200+1,
        "display_step": 100,
        "embedding_size":50,
        "n_input": 50,
        "n_hidden_dense1": 32,
        "n_hidden_dense2": 16,
        "version": "v0",
        "max_grad_norm":100,
        "stddev": 0.01,
        "emb_learning_rate": 0.1,
        "dropout_prob":0.1
    }
    
    net = SDPP(config,n_nodes)
    print(net)
    print(net.get_parameters())

    learning_rate = 1e-4
    optimizers = torch.optim.Adam(net.get_parameters(),lr = learning_rate)
    