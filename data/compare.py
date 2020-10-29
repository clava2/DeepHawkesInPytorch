import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def construct_set(line) -> set:
    edges = line.split(" ")
    result = set()
    for edge in edges:
        result.add(edge)
    return result

def check_same(line1,line2) -> bool:
    if(line1 == line2):
        return True
    parts1 = line1.split("\t")
    parts2 = line2.split("\t")
    if(parts1[0] != parts2[0]):
        return False
    if(parts1[1] != parts2[1]):
        return False
    if(parts1[2] != parts2[2]):
        return False
    if(parts1[3] != parts2[3]):
        return False
    if(parts1[5] != parts2[5]):
        return False
    set1 = construct_set(parts1[4])
    set2 = construct_set(parts2[4])
    if(set1 != set2):
        return False
    return True

def compare_file(filename1,filename2):
    file1 = open(filename1,'r')
    file2 = open(filename2,'r')
    line_count = 0
    while(True):
        line1 = file1.readline()
        line2 = file2.readline()
        if(not check_same(line1,line2)):
            logging.info("line is not the same. line count: %d " % line_count)
            logging.info("line1 length: %d, line1: %s" % (len(line1),line1))
            logging.info("line1 length: %d, line2: %s" % (len(line2),line2))
            return False
        if((not line1) and (not line2)):
            return True
        line_count += 1

if __name__ == "__main__":
    if(compare_file("data/original/shortestpath_val.txt","/bigdisk/zc/weibo/DeepHawkes_py3.6/dataset/shortestpath_val.txt")):
        logging.info("file is the same")
    else:
        logging.info("file is not the same")