import numpy as np
import yaml
import pickle
import os

LABEL_NUM = 0

class IndexDict:
    def __init__(self,original_ids):
        self.original_to_new = {}
        self.new_to_original = []
        count = 0
        for i in original_ids:
            new = self.original_to_new.get(i,count)
            if new == count:
                self.original_to_new[i] = count
                count += 1
                self.new_to_original.append(i)
    def new(self,original):
        if(type(original) is int):
            return self.original_to_new[original]
        else:
            if(type(original[0]) is int):
                return [self.original_to_new[i] for i in original]
            else:
                return [[self.original_to_new[i] for i in l] for l in original]
    
    def original(self,new):
        if(type(new) is int):
            return self.new_to_original[new]
        else:
            if(type(new[0]) is int):
                return [self.new_to_original[i] for i in new]
            else:
                return [[self.new_to_original[i] for i in l] for l in new]
    def length(self):
        return len(self.new_to_original)


#trainsform the sequence to list
# 读取输入文件中的图结构
def sequence2list(flename):
    # 图
    graphs = {}
    # 打开文件
    with open(flename, 'r') as f:
        for line in f:
            global walks
            # 使用\t分隔
            walks = line.strip().split('\t')
            # walks[0]是id
            graphs[walks[0]] = []
            for i in range(1, len(walks)):
                # start 节点
                s = walks[i].split(":")[0]
                # to 节点
                t = walks[i].split(":")[1]
                graphs[walks[0]].append([[int(xx) for xx in s.split(",")],int(t)])
    # 得到的图结构
    return graphs

# 读取标签和大小
def read_labelANDsize(filename):
    # 标签 id->labels
    labels = {}
    # 尺寸 id->size
    sizes = {}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            # 标签是最后一个
            labels[profile[0]] = profile[-1]
            # 大小是第三个
            sizes[profile[0]] = int(profile[3])
    return labels,sizes

# 读取原始id
def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            # print graph,walk
            # print "walk",walk[0],walk[1]
            for i in walk[0]:
                original_ids.add(i)
    print("length of original isd:",len(original_ids))
    return original_ids

# 写入数据
def write_XYSIZE_data(graphs,labels,sizes,LEN_SEQUENCE,NUM_SEQUENCE,index,filename):
    """
    graphs:
    labels:
    sizes:
    LEN_SEQUENCE:
    NUM_SEQUENCE:
    index:
    filename:
    """
    #get the x,y,and size  data
    blank_template = []
    for i in range(LEN_SEQUENCE):
        blank_template.append(index.new(-1))
    print("blank_template",len(blank_template),blank_template)
    x_data = []
    y_data = []
    sz_data = []
    time_data = []
    rnn_index = []
    for key,graph in graphs.items():
        # print key
        label = labels[key].split()
        y = int(label[LABEL_NUM])
        temp = []
        temp_time = []
        temp_index = []
        count = 0
        size_temp = len(graph)
        if size_temp !=  sizes[key]:
        	print(size_temp,sizes[key])
        for walk in graph:
            # print walk
            temp_walk = []
            walk_time = walk[1]
            temp_time.append(walk_time)
            temp_index.append(len(walk[0]))
            # print walk
            for w in walk[0]:
                temp_walk.append(index.new(w))
            while len(temp_walk) <LEN_SEQUENCE:
                temp_walk.append(index.new(-1))
            # print temp_walk
            temp.append(temp_walk)
            count +=1
        x_data.append(temp)
        y_data.append(np.log(y+1.0)/np.log(2.0))
        sz_data.append(size_temp)
        time_data.append(temp_time)
        rnn_index.append(temp_index)
    #print x_data
#    print(len(x_data),len(x_data[0]),len(x_data[0][0]))
    pickle.dump((x_data, y_data, sz_data, time_data,rnn_index,index.length()), open(filename,'wb'))

def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        # print cascadeID,sizes[cascadeID]
        max_size = max(max_size,sizes[cascadeID])
    print("max_size",max_size)
    return max_size
def get_max_length(graphs):
 #   max_overlap = 0
    len_sequence = 0
    max_num = 0
    for cascadeID in graphs:
        max_num = max(max_num,len(graphs[cascadeID]))
        for sequence in graphs[cascadeID]:
            len_sequence = max(len_sequence,len(sequence[0]))
    print("max_num:",max_num)
    return len_sequence

if __name__ == "__main__":
    ## 读取配置文件
    config_path = "config/main.yaml"
    config = open(config_path,'r')
    file_data = config.read()
    config = yaml.load(file_data)
    preprocess_config = config["preprocess"]

    # 读取训练、验证和测试集中的图结构
    graphs_train = sequence2list(os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path-data"]["train"]))
    graphs_val = sequence2list(os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path-data"]["val"]))
    graphs_test = sequence2list(os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path-data"]["test"]))

    # 读取训练、验证和测试集中的标签和大小
    labels_train ,sizes_train = read_labelANDsize(os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["train"]))
    labels_val , sizes_val = read_labelANDsize(os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["val"]))
    labels_test , sizes_test = read_labelANDsize(os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["test"]))

    # 序列数量为训练集、验证集合测试集中的最大值
    NUM_SEQUENCE = max(get_maxsize(sizes_train),get_maxsize(sizes_val),get_maxsize(sizes_test))
    print("Number of sequence:",NUM_SEQUENCE)
    # 向20取整
    NUM_SEQUENCE = (NUM_SEQUENCE/20+1)*20
    print(NUM_SEQUENCE)

    # 训练集的路径长度
    LEN_SEQUENCE_train = get_max_length(graphs_train)
    # 验证集的路径长度
    LEN_SEQUENCE_val = get_max_length(graphs_val)
    # 测试集的路径长度
    LEN_SEQUENCE_test = get_max_length(graphs_test)
    # 路径长度
    LEN_SEQUENCE = max(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test)
    # LEN_SEQUENCE 为每个级联的最大跳数
    print("\nlength of sequence:",LEN_SEQUENCE)
    print(LEN_SEQUENCE_train,LEN_SEQUENCE_val,LEN_SEQUENCE_test)

    # 获得原始id集合
    original_ids = get_original_ids(graphs_train)\
                    .union(get_original_ids(graphs_val))\
                    .union(get_original_ids(graphs_test))
    
    # 添加特殊节点-1
    original_ids.add(-1)
    print("lenth of original_ids:",len(original_ids))
    index = IndexDict(original_ids)
    pickle.dump((index.new_to_original), open("idmap.pkl", 'wb'))
    #write the x,y,and size  data
    write_XYSIZE_data(graphs_train, labels_train, sizes_train,LEN_SEQUENCE,NUM_SEQUENCE,index, 
                        os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["train"]))
    write_XYSIZE_data(graphs_val, labels_val, sizes_val,LEN_SEQUENCE,NUM_SEQUENCE,index, 
                        os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["val"]))
    write_XYSIZE_data(graphs_test, labels_test, sizes_test,LEN_SEQUENCE,NUM_SEQUENCE,index,
                        os.path.join(preprocess_config["main-data-path"],preprocess_config["pickle"]["test"]))

    # write the node information
    pickle.dump((len(original_ids),int(NUM_SEQUENCE),LEN_SEQUENCE), 
                    open(os.path.join(preprocess_config["main-data-path"],preprocess_config["information"]),'wb'))