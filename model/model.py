import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPP():
    def __init__(self,config,n_nodes):
        self.device = torch.device("cuda")
        self.pred = torch.tensor([1,1,1,1,1])
        self.y    = torch.tensor([1,2,3,4,5])
        self.loss = torch.sub(self.pred,self.y).pow(2).sum()
        self.error = torch.sub(self.pred,self.y).pow(2).sum()
        self.n_sequences = config["n_sequences"]
        self.n_hidden_gru = config["n_hidden_gru"]
        self.n_hidden_dense1 = config["n_hidden_dense1"]
        self.n_hidden_dense2 = config["n_hidden_dense2"]
        self.embedding_size = config["embedding_size"]
        self.n_time_interval = config["n_time_interval"]
        self.dropout_prob = config["dropout_prob"]
        self.n_input = config["n_input"]
        self.n_steps = config["n_steps"]
        self.batch_size = config["batch_size"]
        self.n_nodes = n_nodes
        self.weights = {
            "dense1": torch.rand(2 * self.n_hidden_gru     , self.n_hidden_dense1    , requires_grad = True,device = self.device),
            "dense2": torch.rand(    self.n_hidden_dense1, self.n_hidden_dense2, requires_grad = True,device = self.device),
            "out": torch.rand(    self.n_hidden_dense2, 1                   , requires_grad = True,device=self.device)
        }
        self.biases = {
            "dense1": torch.rand(self.n_hidden_dense1, requires_grad = True,device = self.device),
            "dense2": torch.rand(self.n_hidden_dense2, requires_grad = True,device = self.device),
            "out"   : torch.rand(1                   , requires_grad = True,device = self.device)
        }
        self.embedding = torch.rand(self.n_nodes          , self.embedding_size,  requires_grad = True,device = self.device)
        self.time_weight = torch.rand(self.n_time_interval, requires_grad = True,device = self.device)
        self.parameters = [self.weights["dense1"],self.weights["dense2"],self.weights["out"],
                            self.biases["dense1"],self.biases["dense2"],self.biases["out"],
                            self.embedding,self.time_weight]
        self.gru = torch.nn.GRU(self.n_input,2 * self.n_hidden_gru)
        self.gru = self.gru.to(self.device)
        self.activation = F.relu

    def forward(self,x,rnn_index,time_interval_index,x_indict):
        x = torch.tensor(x,device=self.device)
        rnn_index = torch.tensor(rnn_index,dtype=torch.float32,device = self.device)
        time_interval_index = torch.tensor(time_interval_index,dtype=torch.float32,device = self.device)
        x_vector = F.dropout(F.embedding(x,self.embedding),self.dropout_prob)
        # (total number of sequence, n_steps, n_input)
        x_vector = torch.transpose(x_vector,1,0)
        # (n_steps,total number of sequence,n_input)
        x_vector = torch.reshape(x_vector,(-1,self.n_input))
        # (n_steps*total_number of sequence,n_input)

        x_vector = torch.split(x_vector,self.n_steps)

        x_vector = torch.stack(x_vector)
        # print(type(x_vector[0]))
        # print(x_vector[0])
        outputs,states = self.gru(x_vector)

        hidden_states = torch.transpose(outputs,1,0)
        # (total number of sequence,n_steps,n_hidden_gru)

        hidden_states = torch.reshape(hidden_states,[-1,2 * self.n_hidden_gru])
        # (total number of sequence * n_step,2 * n_hidden_gru)

        rnn_index = torch.reshape(rnn_index,[-1,1])
        # (total number of sequence* n_step,1)

        hidden_states = torch.mul(rnn_index,hidden_states)
        # (total number of sequence* n_step, 2 * n_hidden_gru)

        hidden_states = torch.reshape(hidden_states,[-1,self.n_steps,2 * self.n_hidden_gru])
        # (total numbe of sequence,n_step,2*n_hidden_gru)

        hidden_states = torch.sum(hidden_states,1)
        # (total number of sequence,2 * n_hidden_gru)

        time_weight = torch.reshape(self.time_weight,[-1,1])
        # (n_time_interval,1)

        # (time interval index) * (total numbe of sequence,n_time_interval)
        time_weight = torch.matmul(time_interval_index,time_weight)
        # (total number of sequence,1)

        hidden_graph_value = torch.mul(time_weight,hidden_states)
        # (total number of sequence, 2*n_hidden_gru)

        hidden_graph_value = torch.reshape(hidden_graph_value,[-1])
        # (total number of sequence*2*n_hidden_gru)

        x_indict = torch.tensor(x_indict,device = self.device)
        hidden_graph = torch.sparse_coo_tensor(indices = torch.transpose(x_indict,0,1), values = hidden_graph_value, 
                                                size = [self.batch_size,self.n_sequences, 2 * self.n_hidden_gru],device = self.device)

        # hidden_graph = torch.sum(hidden_graph_value,[1])
        hidden_graph = hidden_graph.to_dense().sum(1)
        # (self.batch_size,2 * self.n_hidden_gru)


        dense1 = self.activation(torch.add(torch.matmul(hidden_graph,self.weights["dense1"]),self.biases["dense1"]))
        dense2 = self.activation(torch.add(torch.matmul(dense1,self.weights["dense2"]),self.biases["dense2"]))
        pred = self.activation(torch.add(torch.matmul(dense2,self.weights["out"]),self.biases["out"]))
        self.pred = pred.reshape((self.batch_size))
    def get_loss(self,y):
        y =  torch.tensor(y,dtype=torch.float32,device = self.device)
        result = (y - self.pred).pow(2).sum()
        result.contiguous()
        for param in self.get_parameters():
            param.contiguous()
        return result
    
    def get_parameters(self):
        return self.parameters



# class SDPP(nn.Module):
#     def __init__(self,config,n_nodes):
#         super(SDPP,self).__init__()
#         self.pred = torch.tensor([1,1,1,1,1])
#         self.y    = torch.tensor([1,2,3,4,5])
#         self.loss = torch.sub(self.pred,self.y).pow(2).sum()
#         self.error = torch.sub(self.pred,self.y).pow(2).sum()
#         self.n_sequences = config["n_sequences"]
#         self.n_hidden_gru = config["n_hidden_gru"]
#         self.n_hidden_dense1 = config["n_hidden_dense1"]
#         self.n_hidden_dense2 = config["n_hidden_dense2"]
#         self.embedding_size = config["embedding_size"]
#         self.n_time_interval = config["n_time_interval"]
#         self.n_nodes = n_nodes
#         self.weights = {
#             "dense1": torch.rand(2 * self.n_hidden_gru,    self.n_hidden_dense1, requires_grad = True),
#             "dense2": torch.rand(    self.n_hidden_dense1, self.n_hidden_dense2, requires_grad = True),
#             "out":    torch.rand(    self.n_hidden_dense2, 1                   , requires_grad = True)
#         }
#         self.biases = {
#             "dense1": torch.rand(self.n_hidden_dense2, requires_grad = True),
#             "dense2": torch.rand(self.n_hidden_dense2, requires_grad = True),
#             "out"   : torch.rand(1                   , requires_grad = True)
#         }
#         self.embedding = torch.rand(self.n_nodes          , self.embedding_size,  requires_grad = True)
#         self.time_weight = torch.rand(self.n_time_interval, requires_grad = True)
    
#     def build_input(self):
#         self.x = torch.tensor([])
    
#     def build_model(self):
#         # self.embedding = nn.Embedding(self.n_nodes,self.embedding_size)
#         self.dropout = nn.Dropout(self.dropout_prob)

#         # (total number of sequence,n_steps,n_input)
        
#     def forward(self,x):
#         x_vector = self.dropout(self.embedding(x))
#         # (total number of sequence, n_steps, n_input)
#         x_vector = torch.transpose(x_vector,1,0)
#         # (n_steps,total number of sequence,n_input)
#         x_vector = torch.reshape(x_vector,(-1,self.n_inputs))
#         # (n_steps*total_number of sequence,n_input)

#         x_vector = torch.split(x_vector,self.n_steps)

#         outputs,states = self.gru(x_vector)

#         hidden_states = torch.transpose(torch.stack(outputs),1,0)
#         # (total number of sequence,n_steps,n_hidden_gru)

#         hidden_states = torch.reshape(hidden_states,[-1,2 * self.n_hidden_gru])
#         # (total number of sequence * n_step,2 * n_hidden_gru)

#         rnn_index = torch.reshape(self.rnn_index,[-1,1])
#         # (total number of sequence* n_step,1)

#         hidden_states = torch.mul(rnn_index,hidden_states)
#         # (total number of sequence* n_step, 2 * n_hidden_gru)

#         hidden_states = torch.reshape(hidden_states,[-1,self.n_steps,2 * self.n_hidden_gru])
#         # (total numbe of sequence,n_step,2*n_hidden_gru)

#         hidden_states = torch.sum(hidden_states,1)
#         # (total number of sequence,2 * n_hidden_gru)

#         time_weight = torch.reshape(self.time_weight,[-1,1])
#         # (n_time_interval,1)

#         # (time interval index) * (total numbe of sequence,n_time_interval)
#         time_weight = torch.matmul(self.time_interval_index,time_weight)
#         # (total number of sequence,1)

#         hidden_graph_value = torch.mul(time_weight,hidden_states)
#         # (total number of sequence, 2*n_hidden_gru)

#         hidden_graph_value = torch.reshape(hidden_graph_value,[-1])
#         # (total number of sequence*2*n_hidden_gru)

#         hidden_graph = torch.sum(hidden_graph_value,1)
#         # (self.batch_size,2 * self.n_hidden_gru)


#         dense1 = self.activation(torch.add(torch.matmul(hidden_graph,self.weights["dense1"]),self.biases["dense1"]))
#         dense2 = self.activation(torch.add(torch.matmul(dense1,self.weights["dense2"]),self.biases["dense2"]))
#         pred = self.activation(torch.add(torch.matmul(dense2,self.weights["out"]),self.biases["out"]))

#         print(pred.size())
#         return pred
