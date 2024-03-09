from load_data import dataloader_train
from load_data import dataloader_valid
from load_data import dataloader_test

import numpy as np
import torch
import torch.nn as nn

class Multimodal(nn.Module):
    def __init__(self):
        super(Multimodal, self).__init__()
        #画像データ
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, 3),                #(120, 120, 3) --> (118, 118, 32)
                        nn.MaxPool2d(2),                    #(118, 118, 32) --> (59, 59, 32)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),    #ここをinplace=Falseにしないとloss.backward()がうまくいかない。
                        
                        nn.Conv2d(32, 64, 3),               #(59, 59, 32) --> (57, 57, 64)
                        nn.MaxPool2d(2),                    #(57, 57, 64) --> (28, 28, 64)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),    
                        
                        nn.Conv2d(64, 128, 3),              #(28, 28, 64) --> (26, 26, 128)
                        nn.MaxPool2d(2),                    #(26, 26, 128) --> (13, 13, 128)
                        nn.ReLU(),
                        nn.Dropout(0.25, inplace=False),
                        
                        nn.Flatten(),                       #(13*13*128)
                        nn.Linear(13*13*128, 64),           #(13*13*128) --> (64)
                        nn.ReLU())
        #技データ
        #いったんただの全結合層だけでやってみて、それからLSTMでやってみよう
        self.fc = nn.Sequential(nn.Linear(2, 2),
                                nn.ReLU(),
                                nn.Linear(2, 2),
                               nn.ReLU())
        self.linear = nn.Linear(64+2, 3)
    
    # forwardへの入力はx:画像 s:技
    def forward(self, x, s):
        x = self.cnn(x)
        #print('cnn後のimgのshape:{}'.format(x.shape))
        #s = self.fc(s)
        #print('fc後のmoveのshape:{}'.format(s.shape))
        concat_output = torch.cat((x, s), axis=1)
        y = self.linear(concat_output)
        return y
    
#Multimodalインスタンス作成
model_1 = Multimodal()

# Heの初期化
def init_weights(model):
    if type(model) == nn.Linear or type(model) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(model.weight)
        model.bias.data.fill_(0.0)


# 学習プロセス
def train(model, train_loader, valid_loader, criterion, optimizer, epochs):
    #パラメータの初期化
    model.apply(init_weights)
    model.to(device)
    for epoch in range(epochs):
        losses_train = []
        losses_valid = []

        model.train()
        n_train = 0
        acc_train = 0

        for target, move, img in train_loader:
            n_train += target.size()[0]

            img = img.to(torch.float32)
            img = img.to(device)
            
            move = move.to(torch.float32)
            move = move.to(device)
            
            target = target.to(device)
            
            
            optimizer.zero_grad()
            #print('imgのshapeは:{}'.format(img.shape))
            output = model.forward(img, move)
            #print('outputのtypeは:{}'.format(output.dtype))
            #print('outputのtypeは:{}'.format(type(output)))
            #print('targetのtypeは:{}'.format(type(target)))
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)

            acc_train += pred.eq(target.view_as(pred)).sum().item()
            losses_train.append(loss.tolist())
        
        model.eval()
        n_val = 0
        acc_val = 0

        with torch.no_grad():
            for target, move, img in valid_loader:
                n_val += target.size()[0]

                output = model.forward(img, move)
                pred = output.argmax(dim=1, keepdim=True)

                acc_val += pred.eq(target.view_as(pred)).sum().item()
                losses_valid.append(loss.tolist())

        print("EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Validation [Loss: {:.3f}, Accuracy: {:.3f}]".format(
        epoch,
        np.mean(losses_train),
        acc_train/n_train,
        np.mean(losses_valid),
        acc_val/n_val))

#-----訓練-----#
n_epochs = 10
lr = 0.0005
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=lr)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')       

#パラメータの初期化
model_1.apply(init_weights)

#実行
train(model_1, dataloader_train, dataloader_valid, loss_func, optimizer, n_epochs)