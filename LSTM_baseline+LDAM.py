from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import random

class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class TrainData(Dataset):
    
    def __init__(self, data, lengths, labels):
        # padding
        self.data = pad_sequence(data, batch_first=True)
        self.lengths = lengths
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input = self.data[index]
        length = self.lengths[index]
        label = self.labels[index]
        return input, length, label

def prepare_data(input):
    filename = str(input['stay'])
    df = pd.read_csv(filename)
    # only choose numeric-valued features
    numeric_cols = ['Hours', 'Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',\
            'Glascow coma scale total', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',\
            'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature',\
            'Weight', 'pH']
    df = df[numeric_cols]
    for col in numeric_cols:
        # fill missing data
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(value=0)
    # df.to_csv(filename)
    df_tensor = torch.tensor(df.values)
    return df_tensor

class Model(nn.Module):
    def __init__(self, device, input_size, output_size, hidden_dim, n_layers, dropout):
        super(Model, self).__init__()

        self.device = device
        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, length):
        
        x = x.float()
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        x = pack_padded_sequence(x, length.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_x, (hidden, cell) = self.lstm(x, hidden)
        # x, _ = pad_packed_sequence(packed_x, batch_first=True)

        # Reshaping the hidden/cell state such that it can be fit into the fully connected layer
        hidden = hidden.contiguous().view(-1, self.hidden_dim)

        hidden = self.dropout(hidden)
        hidden = self.fc(hidden)
        hidden = hidden.view(batch_size, -1)

        return hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))
            
        return hidden

class LDAMLoss(nn.Module):
    
    def __init__(self, device, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        self.device = device
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # m_list = torch.cuda.FloatTensor(m_list) torch.from_numpy
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list.to(device)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8).to(self.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        # index_float = index.type(torch.cuda.FloatTensor)
        index_float = index.type(torch.FloatTensor).to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        # print(index_float)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

def train_model(device, model, train_loader, test_loader, lr, weight_decay, n_epochs, batch_size, cls_num_list, oversample=False, loss_type="CE", reweighting=False):
    if reweighting:
        per_cls_weights = torch.FloatTensor(cls_num_list / np.sum(cls_num_list) * len(cls_num_list)).to(device)
    else:
        per_cls_weights = None
    # Define Loss, Optimizer
    if loss_type == "CE":
        criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
    elif loss_type == "LDAM":
        criterion = LDAMLoss(device, cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize tracker for best test metrics
    test_f1_max = 0.0
    test_auc_max = 0.0 

    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        test_loss = 0.0
        num_samples_train = 0
        num_samples_test = 0
        
        # Training Run
        model.train() # prep model for training
        for data, length, target in train_loader:
            data = data.to(device)
            length = length.to(device)
            target = target.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, length)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # update running training loss
            train_loss += loss.item()*data.size(0)
            num_samples_train += data.size(0)

        model.eval() # prep model for evaluation
        for data, length, target in test_loader:
            data = data.to(device)
            length = length.to(device)
            target = target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data, length)
            # calculate the loss
            loss = criterion(output, target)
            # update running test loss 
            test_loss += loss.item()*data.size(0)
            num_samples_test += data.size(0)

        train_loss /= num_samples_train
        test_loss /= num_samples_test
        with torch.no_grad():
            y_train_pred = torch.tensor([])
            y_train_true = torch.tensor([])
            y_train_score = torch.tensor([])
            for data, length, target in train_loader:
                data = data.to(device)
                length = length.to(device)
                output = model(data, length).cpu()
                # normalize logits adn get prob of class 1
                score_batch = F.softmax(output, dim=1)[:, 1]
                _, pred_batch = output.max(1)
                y_train_pred = torch.cat((y_train_pred, pred_batch))
                y_train_score = torch.cat((y_train_score, score_batch))
                y_train_true = torch.cat((y_train_true, target))
            y_train_pred = np.array(y_train_pred)
            y_train_score = np.array(y_train_score)
            acc_train = accuracy_score(y_train_true, y_train_pred)
            f1_train = f1_score(y_train_true, y_train_pred, average='macro')
            auc_train = roc_auc_score(y_train_true, y_train_score)

            y_test_pred = torch.tensor([])
            y_test_true = torch.tensor([])
            y_test_score = torch.tensor([])
            for data, length, target in test_loader:
                data = data.to(device)
                length = length.to(device)
                output = model(data, length).cpu()
                 # normalize logits adn get prob of class 1
                score_batch = F.softmax(output, dim=1)[:, 1]
                _, pred_batch = output.max(1)
                y_test_pred = torch.cat((y_test_pred, pred_batch))
                y_test_score = torch.cat((y_test_score, score_batch))
                y_test_true = torch.cat((y_test_true, target))
            y_test_pred = np.array(y_test_pred)
            y_test_score = np.array(y_test_score)
            acc_test = accuracy_score(y_test_true, y_test_pred)
            f1_test = f1_score(y_test_true, y_test_pred, average='macro')
            auc_test = roc_auc_score(y_test_true, y_test_score)
        print()
        print('Epoch: {} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTrain acc: {:.6f} \tTest acc: {:.6f}\tTrain f1: {:.6f} \tTest f1: {:.6f}\tTrain AUC: {:.6f} \tTest AUC: {:.6f}'.format(
          epoch+1, 
          train_loss,
          test_loss,
          acc_train,
          acc_test,
          f1_train,
          f1_test,
          auc_train,
          auc_test
          ))
        
        # save model if test metrics has increased
        if f1_test >= test_f1_max:
            print('F1 increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
              test_f1_max,
              f1_test))
            if oversample:
                torch.save(model.state_dict(), f'lstm_{loss_type}_with_oversample_f1.pt')
            else:
                torch.save(model.state_dict(), f'lstm_{loss_type}_without_oversample_f1.pt')
            test_f1_max = f1_test
        if auc_test >= test_auc_max:
            print('AUC increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
              test_auc_max,
              auc_test))
            if oversample:
                torch.save(model.state_dict(), f'lstm_{loss_type}_with_oversample_auc.pt')
            else:
                torch.save(model.state_dict(), f'lstm_{loss_type}_without_oversample_auc.pt')
            test_auc_max = auc_test

@torch.no_grad()
def evaluate(saved_model, test_loader):
    model = Model(device='cpu', input_size=15, output_size=2, hidden_dim=32, n_layers=1, dropout=0.2)
    model.eval()
    model.load_state_dict(torch.load(saved_model))
    y_test_pred = torch.tensor([])
    y_test_true = torch.tensor([])
    y_test_score = torch.tensor([])
    for data, length, target in test_loader:
        output = model(data, length)
        # normalize logits adn get prob of class 1
        score_batch = F.softmax(output, dim=1)[:, 1]
        _, pred_batch = output.max(1)
        y_test_pred = torch.cat((y_test_pred, pred_batch))
        y_test_score = torch.cat((y_test_score, score_batch))
        y_test_true = torch.cat((y_test_true, target))
    y_test_pred = np.array(y_test_pred)
    y_test_score = np.array(y_test_score)
    acc_test = accuracy_score(y_test_true, y_test_pred)
    f1_test = f1_score(y_test_true, y_test_pred, average='macro')
    auc_test = roc_auc_score(y_test_true, y_test_score)
    return f1_test, auc_test


def train():
    # read file
    df_train = pd.read_csv('1_train_listfile801010.csv')
    df_test = pd.read_csv('1_test_listfile801010.csv')

    # prepare data
    df_train['tensor'] = df_train.apply(prepare_data, axis=1)
    df_test['tensor'] = df_test.apply(prepare_data, axis=1)

    # prepare data loaders
    data_vectors_train = df_train['tensor'].values
    lengths_train = [x.size(0) for x in data_vectors_train]
    labels_train = df_train['y_true'].values
    cls_num_list = np.unique(labels_train, return_counts=True)[1][::-1]
    train_data = TrainData(data_vectors_train, lengths_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128)
    data_vectors_test = df_test['tensor'].values
    lengths_test = [x.size(0) for x in data_vectors_test]
    labels_test = df_test['y_true'].values
    test_data = TrainData(data_vectors_test, lengths_test, labels_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    # oversampling
    data_vectors_train_0 = [data_vectors_train[i] for i in range(len(data_vectors_train)) if labels_train[i] == 0]
    majority_len = len(data_vectors_train_0)
    data_vectors_train_1 = [data_vectors_train[i] for i in range(len(data_vectors_train)) if labels_train[i] == 1]
    data_vectors_train_1_os = random.choices(data_vectors_train_1, k=majority_len)
    data_vectors_train_os = data_vectors_train_0 + data_vectors_train_1_os
    lengths_train_os = [x.size(0) for x in data_vectors_train_os]
    labels_train_os = np.concatenate((np.zeros(majority_len),np.ones(majority_len)), axis=None).astype('int64')
    cls_num_list_os = np.unique(labels_train_os, return_counts=True)[1][::-1]
    train_data_os = TrainData(data_vectors_train_os, lengths_train_os, labels_train_os)
    train_loader_os = torch.utils.data.DataLoader(train_data_os, batch_size=128, shuffle=True)
    
    # CE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model with hyperparameters
    model = Model(device=device, input_size=15, output_size=2, hidden_dim=32, n_layers=1, dropout=0.2)
    model = model.to(device)
    # Define hyperparameters
    n_epochs = 50
    lr = 5e-3
    batch_size = 128
    weight_decay = 0
    print("Running on LSTM using Cross Entropy loss without oversampling...")
    train_model(device, model, train_loader, test_loader, lr, weight_decay, n_epochs, batch_size, cls_num_list, oversample=False)
    
    # CE+oversampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model with hyperparameters
    model = Model(device=device, input_size=15, output_size=2, hidden_dim=32, n_layers=1, dropout=0.2)
    model = model.to(device)
    # Define hyperparameters
    n_epochs = 50
    lr = 5e-3
    batch_size = 128
    weight_decay = 0
    print("Running on LSTM using Cross Entropy loss with oversampling...")
    train_model(device, model, train_loader_os, test_loader, lr, weight_decay, n_epochs, batch_size, cls_num_list, oversample=True)
    
    # LDAM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model with hyperparameters
    model = Model(device=device, input_size=15, output_size=2, hidden_dim=32, n_layers=1, dropout=0.2)
    model = model.to(device)
    # Define hyperparameters
    n_epochs = 50
    lr = 5e-3
    batch_size = 128
    weight_decay = 0
    print("Running on LSTM using LDAM loss without oversampling...")
    train_model(device, model, train_loader, test_loader, lr, weight_decay, n_epochs, batch_size, cls_num_list, oversample=False, loss_type="LDAM", reweighting=True)
    
    # LDAM+oversampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Instantiate the model with hyperparameters
    model = Model(device=device, input_size=15, output_size=2, hidden_dim=32, n_layers=1, dropout=0.2)
    model = model.to(device)
    # Define hyperparameters
    n_epochs = 50
    lr = 5e-3
    batch_size = 128
    weight_decay = 0
    print("Running on LSTM using LDAM loss with oversampling...")
    train_model(device, model, train_loader_os, test_loader, lr, weight_decay, n_epochs, batch_size, cls_num_list_os, oversample=True, loss_type="LDAM", reweighting=True)

    #evaluate models
    for loss_type in ['CE', 'LDAM']:
        for oversample in ['without', 'with']:
            for metric in ['f1', 'auc']:
                filename = f'lstm_{loss_type}_{oversample}_oversample_{metric}.pt'
                f1_test, auc_test = evaluate(filename, test_loader)
                print(f"Test F1 score and AUC score of LSTM using {loss_type} {oversample} oversample(saved on best {metric}): {f1_test}, {auc_test}")

if __name__ == '__main__':

    train()
