import time
import logging
import yaml
import os

logging.basicConfig(filename='logger.log', level=logging.INFO)

def check_valid_nodes(path):
    return -1 not in set([int(n) for n in path.split(":")[0].split("/")])

def check_line(line,observation_time) -> bool:
    parts = line.split("\t")
    if len(parts) != 5:
        logging.info('wrong format!')
        return False
    n_nodes = int(parts[3])
    path = parts[4].split(" ")
    if n_nodes !=len(path):
        print('wrong number of nodes',n_nodes,len(path))
    observation_path_length = 0
    for p in path:
        if(not check_valid_nodes(p)):
            continue
        if int(p.split(":")[1]) < observation_time:
            observation_path_length += 1
    if observation_path_length <10 or observation_path_length >1000:
        return False
    return True


def gen_cascade_graph(observation_time,
                        pre_times,
                        filename,
                        filename_ctrain,
                        filename_cval,
                        filename_ctest,
                        filename_strain,
                        filename_sval,
                        filename_stest):
    """

    """
    file        = open(filename)
    file_ctrain = open(filename_ctrain,"w")
    file_cval   = open(filename_cval,"w")
    file_ctest  = open(filename_ctest,"w")
    file_strain = open(filename_strain,"w")
    file_sval   = open(filename_sval,"w")
    file_stest  = open(filename_stest,"w")

    # map: id -> message time
    cascades_total = {line.split("\t")[0]: int(line.split("\t")[2]) for line in file if check_line(line,observation_time)}

    # map: id -> type   1: train,  2: val,  3: test
    sorted_message_time = sorted(cascades_total.items(),key = None)
    cascades_type_train = {k:1 for k,v in sorted_message_time[                                           : int(len(cascades_total) * 14.0 / 20.0) + 1]}
    cascades_type_val   = {k:2 for k,v in sorted_message_time[int(len(cascades_total) * 14.0 / 20.0) + 1 : int(len(cascades_total) * 17.0 / 20.0) + 1]}
    cascades_type_test  = {k:3 for k,v in sorted_message_time[int(len(cascades_total) * 17.0 / 20.0) + 1 :                                           ]}
    cascades_type       = {**cascades_type_train, **cascades_type_val, **cascades_type_test}

    file.close()
    file = open(filename,"r")
    # split every line in the file
    parts_of_lines = [line.split("\t") for line in file]
    # filter out lines whose length is not 5
    parts_of_lines = [parts for parts in parts_of_lines if len(parts) == 5]
    # extract parts
    ids = [parts[0] for parts in parts_of_lines]
    paths = [parts[4].split(" ") for parts in parts_of_lines]
    n_nodes = [parts[3] for parts in parts_of_lines]
    msg_times = [parts[2] for parts in parts_of_lines]
    hour = [time.strftime("%H",time.localtime(int(msg_time))) for msg_time in msg_times]
    labels = [[0] * len(pre_times) for i in range(len(parts_of_lines))]
    observation_paths = [[",".join(p.split(":")[0].split("/")) + ":" + p.split(":")[1] for p in path if int(p.split(":")[1]) < observation_time] for path in paths]

    nodes_of_paths_of_parts = [[p.split(":")[0].split("/") for p in path if int(p.split(":")[1]) < observation_time] for path in paths]
    edges_of_parts = [[set([nodes[i-1] + ":" + nodes[i] + ":1" for i in range(1,len(nodes))]) for nodes in path] for path in nodes_of_paths_of_parts]
    edges_of_parts = [set.union(*edges) for edges in edges_of_parts]

    labels = [[str(len([p for p in paths[i] if int(p.split(":")[1]) < pre_times[j]]) - len(observation_paths[i]))] for i in range(len(paths)) for j in range(len(pre_times))]

    strain_lines = [ids[i] + "\t" + "\t".join(observation_paths[i]) for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 1]
    strain_lines = [line + ("" if line[-1] == '\n' else "\n") for line in strain_lines]
    sval_lines = [ids[i] + "\t" + "\t".join(observation_paths[i]) for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 2]
    sval_lines = [line + ("" if line[-1] == '\n' else "\n") for line in sval_lines]
    stest_lines = [ids[i] + "\t" + "\t".join(observation_paths[i]) for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 3]
    stest_lines = [line + ("" if line[-1] == '\n' else "\n") for line in stest_lines]
    ctrain_lines = [ids[i]+"\t"+parts_of_lines[i][1]+"\t"+parts_of_lines[i][2]+"\t"+str(len(observation_paths[i]))+"\t"+" ".join(edges_of_parts[i])+"\t"+" ".join(labels[i])+"\n" for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 1]
    cval_lines = [ids[i]+"\t"+parts_of_lines[i][1]+"\t"+parts_of_lines[i][2]+"\t"+str(len(observation_paths[i]))+"\t"+" ".join(edges_of_parts[i])+"\t"+" ".join(labels[i])+"\n" for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 2]
    ctest_lines = [ids[i]+"\t"+parts_of_lines[i][1]+"\t"+parts_of_lines[i][2]+"\t"+str(len(observation_paths[i]))+"\t"+" ".join(edges_of_parts[i])+"\t"+" ".join(labels[i])+"\n" for i in range(len(parts_of_lines)) if ids[i] in cascades_type and cascades_type[ids[i]] == 3]

    file_ctrain.writelines(ctrain_lines)
    file_cval.writelines(cval_lines)
    file_ctest.writelines(ctest_lines)
    file_strain.writelines(strain_lines)
    file_sval.writelines(sval_lines)
    file_stest.writelines(stest_lines)

    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()

    file.close()

if __name__ =="__main__":
    config_path = "config/main.yaml"
    config = open(config_path,'r')
    file_data = config.read()
    config = yaml.load(file_data)
    preprocess_config = config["gen-sequence"]
    gen_cascade_graph(preprocess_config["observation-time"],
                        preprocess_config["predict-time"],
    os.path.join(preprocess_config["main-data-path"],preprocess_config["cascades"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["train"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["val"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["cascade-data"]["test"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path"]["train"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path"]["val"]),
    os.path.join(preprocess_config["main-data-path"],preprocess_config["shortest-path"]["test"]))