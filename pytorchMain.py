import os
import sys
import torch
import torch.distributed as dist
import traceback
import datetime
import socket
import pandas as pd
import numpy as np
from random import Random
from pytorchMetadata import metadata_parse
import pickle
import json
import tweepy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pylab as plt
import gzip
import shutil
import os
import wget
import numpy as np
import pandas as pd
import datetime
import torch
import urllib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


features_number = 8
class_feature_number=1
class_number=3
torch.set_printoptions(edgeitems=features_number)

def tweeter_api_keys():
    # Authenticate
    CONSUMER_KEY = "gD6y9mcDSwsZiExRpvIppIMnc" #@param {type:"string"}
    CONSUMER_SECRET_KEY = "xe5mY8JQSkhKTbe6sJuQwVYD92sp56qhHnyKeAnBcoDYH1jX0K" #@param {type:"string"}
    ACCESS_TOKEN_KEY = "1366320174555521026-pZ1U5woqL13Prd2m99TL4OscZvnxj1" #@param {type:"string"}
    ACCESS_TOKEN_SECRET_KEY = "94Y6I7uuJ5UrFPYcI8TrptZWYKAu8rYVEnn6flbquILRf" #@param {type:"string"}

    #Creates a JSON Files with the API credentials
    with open('api_keys.json', 'w') as outfile:
        json.dump({
        "consumer_key":CONSUMER_KEY,
        "consumer_secret":CONSUMER_SECRET_KEY,
        "access_token":ACCESS_TOKEN_KEY,
        "access_token_secret": ACCESS_TOKEN_SECRET_KEY
         }, outfile)


class RangeNormalize(object):
    def __init__(self,
                 min_val,
                 max_val):
        self.min_val = min_val
        self.max_val = max_val


    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val - a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset_hydrate_parse(dataset):
    # root_data='/s/red/a/nobackup/cwc-ro/shadim/clean-dataset.tsv'

    size = dist.get_world_size()
    bsz = int(128 / float(size))
    bsz=100
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz

def all_gather_new(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """

    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage)

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()])
    size_list = [torch.LongTensor([0]) for _ in range(world_size)]

    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,))
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    # data_list = []
    data_list=torch.empty(0,features_number);
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.numpy().tobytes()[:size]
        data_list=torch.cat((data_list,pickle.loads(buffer)),dim=0)

    return data_list



def preprocess(rank, world_size):
    dis = 0

    # partitions_fraction = partition_dataset_hydrate_parse()
    # group all ranks
    ranks = list(range(int(world_size)))
    group = dist.new_group(ranks=ranks)

    '''
    filter_tweets_language()
    '''
    # dataset_URL = "https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/2021-04-18/2021-04-18_clean-dataset.tsv.gz?raw=true"  # @param {type:"string"}
    #
    # # Downloads the dataset (compressed in a GZ format)
    # # !wget dataset_URL -O clean-dataset.tsv.gz
    # wget.download(dataset_URL, out='clean-dataset.tsv.gz')
    #
    # # Unzips the dataset and gets the TSV dataset
    # with gzip.open('clean-dataset.tsv.gz', 'rb') as f_in:
    #     with open('clean-dataset.tsv', 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    #
    # tsvreader = pd.read_csv("clean-dataset.tsv", delimiter="\t",
    #                         dtype={0: 'int64', 1: 'string', 2: 'string', 3: 'string', 4: 'string'})
    # tsv_english = tsvreader[tsvreader['lang'] == 'en']
    # tsvreader_tensor = torch.as_tensor(np.array(tsv_english['tweet_id']), dtype=torch.int64)
    #
    # dataset_URL = "https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/2021-04-19/2021-04-19_clean-dataset.tsv.gz?raw=true"  # @param {type:"string"}
    #
    # # Downloads the dataset (compressed in a GZ format)
    # # !wget dataset_URL -O clean-dataset.tsv.gz
    # wget.download(dataset_URL, out='clean-dataset.tsv.gz')
    #
    # # Unzips the dataset and gets the TSV dataset
    # with gzip.open('clean-dataset.tsv.gz', 'rb') as f_in:
    #     with open('clean-dataset.tsv', 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    #
    # tsvreader = pd.read_csv("clean-dataset.tsv", delimiter="\t",
    #                         dtype={0: 'int64', 1: 'string', 2: 'string', 3: 'string', 4: 'string'})
    # tsv_english = tsvreader[tsvreader['lang'] == 'en']
    # tsvreader_tensor=torch.cat((tsvreader_tensor,torch.as_tensor(np.array(tsv_english['tweet_id']), dtype=torch.int64)),dim=0)

    tsvreader = pd.read_csv("full_data_eng.csv", dtype={0: 'int64'})
    tsvreader_tensor = torch.as_tensor(np.array(tsvreader["0"]), dtype=torch.int64)
    tsvreader_tensor=tsvreader_tensor[10000000:15000000]
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset_hydrate_parse(tsvreader_tensor)
    num_batches = np.ceil(len(train_set.dataset) / float(bsz))

    keyfile="api_keys.json"
    with open(keyfile) as f:
        keys = json.load(f)
    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, retry_delay=60 * 3, retry_count=5,
                     retry_errors=set([401, 404, 500, 503]), wait_on_rate_limit_notify=True)
    if api.verify_credentials() == False:
        print("Your twitter api credentials are invalid")
        sys.exit()
    else:
        print("Your twitter api credentials are valid.")

    all_data_processed = torch.empty(0, features_number);
    len_data_processed=0
    loop_i=0
    for i, (data) in enumerate(train_set):
        loop_i+=1
        len_data_processed+=len(data)
        input_data = metadata_parse(data, api, 'p')
        gathered_tensors=all_gather_new(input_data)
        all_data_processed=torch.cat((all_data_processed,gathered_tensors),dim=0)
        print("all data size:" + str(len(all_data_processed)))
    print("length of data which are processed: "+str(len_data_processed))
    print("length of loop of precessing of data: "+str(loop_i))
    print("*****************************************************************************************************")
    return  all_data_processed

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

def maxValue_columnWise(x):
    return x.max(0, keepdim=True)[0]

# sleep_on_rate_limit=False
class Net(nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()
    #
    #     self.conv1 = nn.Conv1d(1, 5, 2, stride=1)
    #     self.conv2 = nn.Conv1d(5, 10, 2, stride=1)
    #     self.fc1 = nn.Linear((features_number-4-class_feature_number)*10, 100)
    #     self.fc2 = nn.Linear(100, 3)
    #
    # def forward(self, x):
    #     x = x.unsqueeze(1)
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool1d(x, 2, 1)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool1d(x, 2, 1)
    #     x = x.view(-1, (features_number-4-class_feature_number)*10)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return x

    # def __init__(self):
    #     super(Net, self).__init__()
    #
    #     self.conv1 = nn.Conv1d(1, 5, 2, stride=1)
    #     self.conv2 = nn.Conv1d(5, 10, 2, stride=1)
    #     self.conv3 = nn.Conv1d(10, 50, 2, stride=1)
    #     self.fc1 = nn.Linear(1*50, 100)
    #     self.fc2 = nn.Linear(100, 3)
    #
    # def forward(self, x):
    #     x = x.unsqueeze(1)
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool1d(x, 2, 1)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool1d(x, 2, 1)
    #     x = F.relu(self.conv3(x))
    #     x = F.max_pool1d(x, 2, 1)
    #     x = x.view(-1, 1*50)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 5, 2, stride=1)
        self.conv2 = nn.Conv1d(1, 5, 3, stride=1)
        self.conv3 = nn.Conv1d(1, 5, 4, stride=1)
        self.fc1 = nn.Linear(12*5, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool1d(x1, 2, 1)
        x2 = F.relu(self.conv2(x))
        x2 = F.max_pool1d(x2, 2, 1)
        x3 = F.relu(self.conv3(x))
        x3 = F.max_pool1d(x3, 2, 1)
        x4=torch.cat((x1,x2,x3),2)
        x5 = x4.view(-1, 12*5)
        x6 = F.relu(self.fc1(x5))
        x7 = self.fc2(x6)
        return x7

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

""" Partitioning MNIST """
def partition_dataset(dataset):

    size = dist.get_world_size()
    bsz = int(1280 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

""" Distributed Synchronous SGD Example """
def training(rank, size,dataset):
    loss_df = pd.DataFrame()
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(dataset)
    model = nn.parallel.DistributedDataParallel(Net()).double()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)
    weights =[1.3,4,40]
    class_weights = torch.DoubleTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    num_batches = np.ceil(len(train_set.dataset) / float(bsz))
    best_loss = float("inf")
    training_epochs=100
    for epoch in range(training_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        lr_adapt=5
        printProgressBar(0, len(train_set), prefix='Progress:', suffix='Complete', length=50)
        for i, (batch) in enumerate(train_set):
            start_time2 = time.time()
            data=batch[:,0:features_number-class_feature_number]
            # target=batch[:,features_number-class_feature_number:features_number]
            target = batch[:, features_number - class_feature_number]
            targetLen = len(target)
            target.reshape(targetLen)
            target=target.type(torch.LongTensor)
            # lr_adapt=5/((epoch*i)+1)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_adapt
            optimizer.zero_grad()
            output = model(data.double())
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            # print("batch" + str(i))
            printProgressBar(i + 1, len(train_set), prefix='Progress:', suffix='Complete', length=50)
        print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', epoch_loss / num_batches)
        loss_df = loss_df.append([epoch_loss / num_batches])
        # print("loss_df")
        # print(loss_df)
        print("--- %s seconds ---" % (time.time() - start_time))
    # Create dataframe
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1,training_epochs+1),loss_df)
    fig.savefig('training_loss_epochs.png')
    if dist.get_rank() == 0 and epoch_loss / num_batches < best_loss:
        best_loss = epoch_loss / num_batches
        torch.save(model.state_dict(), "best_model.pth")


    return model

""" Distributed Synchronous SGD Example """
def testing(rank, size,dataset,model):

    torch.manual_seed(1234)
    test_set, bsz = partition_dataset(dataset)
    weights = [1.3, 4, 40]
    class_weights = torch.DoubleTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.CrossEntropyLoss()
    num_batches = np.ceil(len(test_set.dataset) / float(bsz))
    epoch_loss = 0.0
    all_test_target=torch.empty(0)
    all_test_predicted_labels = torch.empty(0)
    all_test_predicted=torch.empty((0,class_number))
    total=len(dataset)
    correct=0
    classes=list(range(0,class_number))
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    for i, (batch) in enumerate(test_set):

        data = batch[:, 0:features_number - class_feature_number]
        # target = batch[:, features_number - class_feature_number:features_number]
        target = batch[:, features_number - class_feature_number]
        targetLen=len(target)
        target.reshape(targetLen)
        target = target.type(torch.LongTensor)
        output = model(data.double())
        predicted_labels = torch.max(output, 1).indices
        all_test_target=torch.cat((all_test_target,target),dim=0)
        all_test_predicted=torch.cat((all_test_predicted,output),dim=0)
        all_test_predicted_labels=torch.cat((all_test_predicted_labels,predicted_labels),dim=0)
        loss = criterion(output, target)
        epoch_loss += loss.item()
        # total += target.size(0)



    correct += (all_test_predicted_labels == all_test_target).sum().item()
    for label, prediction in zip(all_test_target, all_test_predicted_labels):
        if label == prediction:
            correct_pred[int(label)] += 1
        total_pred[int(label)] += 1
    print('Accuracy of the network on the %d tweets: %d %%' % (total, 100 * correct / total))

    for classname, correct_count in correct_pred.items():
        if(total_pred[classname]==0):
            accuracy=0
        else :
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:d} is: {:.1f} %".format(classname,
                                                             accuracy))
    # print('Rank ', dist.get_rank(), ', test loss ', epoch_loss / num_batches)
    x_np = all_test_predicted.detach().numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('test_predicted_data.csv')
    x_np = all_test_target.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('test_target_data.csv')
    x_np = dataset.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('test_features_data.csv')

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(class_number):
        fpr[i], tpr[i], thresh[i] = roc_curve(all_test_target.detach(), all_test_predicted[:, i].detach(), pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass_ROC.png', dpi=300)

def classify_retweet(retweet_count):
    if(retweet_count==0):
        return 0
    elif(0<retweet_count<20):
        return 1
    else:
        return 2

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'hartford'
    os.environ['MASTER_PORT'] = '31101'

    # initialize the process group
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://hartford:31102',
                            timeout=datetime.timedelta(weeks=120))

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


if __name__ == '__main__':
    try:
        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname() + ": Setup completed!")
        start_time = time.time()
        # tweeter_api_keys()
        # dataset=preprocess(sys.argv[1], sys.argv[2])
        # x_np = dataset.numpy()
        # x_df = pd.DataFrame(x_np)
        # x_df.to_csv('original_preprocessing_data_big2.csv')
        # print("--- %s seconds ---" % (time.time() - start_time))
        # rn = RangeNormalize(0, 1)
        # for feature_ind in range(0,features_number-class_feature_number):
        #     dataset[:, feature_ind] = (rn(dataset[:, feature_ind]))
        # x_np = dataset.numpy()
        # x_df = pd.DataFrame(x_np)
        # x_df.to_csv('normalized_preprocessing_data_big2.csv')
        # x = dataset['7'].value_counts()
        # print(x)
        dataset=pd.read_csv("normalized_v4_small_SMOTE_train.csv")
        # dataset['7']=dataset['7'].apply(lambda x: classify_retweet(x))

        dataset = torch.as_tensor(np.array(dataset), dtype=torch.float64)
        dataset=dataset[:,-8:]

        data_size=len(dataset)
        train_size=int(0.8*data_size)
        print("start training")
        model=training(sys.argv[1], sys.argv[2],dataset[0:train_size,:])
        print("start testing")
        testing(sys.argv[1], sys.argv[2], dataset[train_size:data_size - 1, :],model)

    except Exception as e:
        traceback.print_exc()
        sys.exit(3)
