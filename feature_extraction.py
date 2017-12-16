from extract_data_2 import main
import torch
import torchvision
from torchvision.models.vgg import model_urls
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


use_gpu = torch.cuda.is_available()

def VGG19_pre_trained():
    # model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
    pre_model = torchvision.models.vgg19_bn(pretrained=True)
    if use_gpu:
        pre_model = pre_model.cuda()
    # for param in pre_model.parameters():
    #     param.requires_grad = False
    layers = list(pre_model.classifier.children())[:-1]
    new_classifier = nn.Sequential(*layers)
    pre_model.classifier = new_classifier
    return pre_model

def forward(X0,X1, pre_model):
    if use_gpu:
        X0 = Variable(torch.from_numpy(X0).cuda())
        X1 = Variable(torch.from_numpy(X1).cuda())
    else:
        X0 = Variable(torch.from_numpy(X0))
        X1 = Variable(torch.from_numpy(X1))
    print X0.size()
    X0_features = pre_model(X0)
    X1_features = pre_model(X1)
    return X0_features, X1_features

def get_data():
    train, validation = main()
    return train, validation


def train_model(pre_model, X0, X1,y):
    criterion = nn.MarginRankingLoss(margin=0.01)
    optimizer = optim.SGD(pre_model.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    loss = criterion(X0, X1, y)
    print loss
    loss.backward()
    optimizer.step()
    

if __name__ == "__main__":
    train, validation = get_data()
    X0, X1, y = train
    print X0.shape
    X0 = np.transpose(X0, (0, 3, 1,2))
    print X0.shape
    X1 = np.transpose(X1, (0, 3, 1,2))
    print X1.shape
    pre_model = VGG19_pre_trained()
    epoch = 2 
    for i in range(epoch):
        #
        X0_batch = None
        X1_batch = None
        y_batch = None
        indices = np.random.choice(np.arange(X0.shape[0]), 50)
        X0_batch = X0[indices]
        X1_batch = X1[indices]
        y_batch = y[indices]
        fx0, fx1 = forward(X0_batch,X1_batch, pre_model)
        train_model(pre_model, fx0,fx1, y)



