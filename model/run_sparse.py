import pickle
import yaml
import os
import sys
import numpy as np
import math
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from model import SDPP

def get_batch(x,y,sz,time,rnn_index,n_time_interval,step,config,batch_size = 128):
    batch_y = np.zeros(shape = (batch_size,1))
    batch_x = []
    batch_x_indict = []
    batch_time_interval_index = []
    batch_rnn_index = []

    start = step * batch_size % len(x)

    for i in range(batch_size):
        id = (i + start) % len(x)
        batch_y[i,0] = y[id]
        for j in range(sz[id]):
            batch_x.append(x[id][j])
            temp_time = np.zeros(shape = (n_time_interval))
            k = int(math.floor(time[id][j] / config["time_interval"]))
            temp_time[k] = 1
            batch_time_interval_index.append(temp_time)

            temp_rnn = np.zeros(shape = (config["n_steps"]))
            if rnn_index[id][j] - 1 >= 0:
                temp_rnn[rnn_index[id][j] - 1] = 1
            batch_rnn_index.append(temp_rnn)
            for k in range(2*config["n_hidden_gru"]):
                batch_x_indict.append([i,j,k])
    return batch_x,batch_x_indict,batch_y,batch_time_interval_index,batch_rnn_index



if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
        "learning_rate": 0.005,
        "sequence_batch_size": 20,
        "batch_size": 32,
        "n_hidden_gru": 32,
        "l1": 5e-5,
        "l2": 0.05,
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
        "emb_learning_rate": 0.0005,
        "dropout_prob": 0.8
    }


    print("dropout prob: %f, l2: %f, learning rate: %f, emb learning rate: %f" % (config["dropout_prob"],config["l2"],config["learning_rate"],config["emb_learning_rate"]))


    training_iters = config["training_iters"]
    batch_size = config["batch_size"]
    display_step = min(config["display_step"],len(sz_train)/batch_size)

    version = config["version"]
    x_train, y_train, sz_train,time_train,rnn_index_train, vocabulary_size = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["train"]),'rb'))
    x_test, y_test, sz_test,time_test,rnn_index_test, _ = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["test"]),'rb'))
    x_val, y_val, sz_val, time_val,rnn_index_val,_ = pickle.load(open(os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["val"]),'rb'))


    start = time.time()
    model = SDPP(config,n_nodes)

    step = 0
    best_val_loss = 1000
    best_test_loss = 1000

    train_loss = []
    max_try = 10
    patience = max_try

    optimizer = torch.optim.Adam(model.get_parameters(),lr = 0.001)

    writer = SummaryWriter()

    while(step * batch_size < training_iters):
        batch_x,batch_x_indict,batch_y,batch_time_interval_index,batch_rnn_index = get_batch(x_train,y_train,sz_train,time_train,rnn_index_train,config["n_time_interval"],step,config,batch_size=batch_size)
        model.forward(batch_x,batch_rnn_index,batch_time_interval_index,batch_x_indict)
        loss = model.get_loss(batch_y)
        print("step: %d, loss: %d" % (step,loss.item()))
        writer.add_scalar("Loss/train",loss,step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1