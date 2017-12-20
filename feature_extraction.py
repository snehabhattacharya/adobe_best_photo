from extract_data import main
import torch
import torchvision
from torchvision.models.vgg import model_urls
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim


use_gpu = torch.cuda.is_available()
criterion = nn.MarginRankingLoss(margin=0.01, size_average=False)
def VGG19_pre_trained():
    # model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
    pre_model = torchvision.models.vgg19_bn(pretrained=True)
    for param in pre_model.features.parameters():
        param.requires_grad = False
    layers = list(pre_model.classifier.children())[:-1]
    new_last_layer = nn.Linear(4096, 1)
    new_tanh = nn.Sigmoid()
    layers += [new_last_layer]
    layers += [new_tanh]
    new_classifier = nn.Sequential(*layers)
    pre_model.classifier = new_classifier
    if use_gpu:
	pre_model = pre_model.cuda()
    print list(pre_model.classifier.children())
    return pre_model

def get_loss(x0, x1, y,margin):
    _output = x0.clone()
    _output.add_(-1, x1)
    _output.mul_(-y[:,None])
    print _output, y
    print _output.size(), y.size()
    _output.add_(margin)
    _output.clamp_(min=0)
    output = _output.sum()
    return output

    
    
    

def forward(X0,X1, pre_model):
    if use_gpu:
        
        X0 = Variable(torch.from_numpy(X0).cuda())
        X1 = Variable(torch.from_numpy(X1).cuda())
    else:
        X0 = Variable(torch.from_numpy(X0))
        X1 = Variable(torch.from_numpy(X1))
    X0_features = pre_model(X0)
    X1_features = pre_model(X1)
    return X0_features, X1_features

def get_data():
    train, validation = main()
    return train, validation


def train_model(pre_model, X0, X1,y, optimizer):
    #X0 = X0.data
    #X1 = X1.data    
    y = Variable(torch.from_numpy(y).type(torch.FloatTensor).cuda())
    #print y.size(), X0.size()   
    #criterion = nn.MarginRankingLoss(margin=0.01)
    # optimizer = optim.SGD(pre_model.classifier[6].parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()
    loss = criterion(X0, X1, y[:,None] )
    # print loss
    loss.backward()
    optimizer.step()
    return pre_model, loss

if __name__ == "__main__":
    train, validation = get_data()
    X0, X1, y = train
    print X0.shape
    X0 = np.transpose(X0, (0, 3, 1,2))
    print X0.shape
    X1 = np.transpose(X1, (0, 3, 1,2))
    print X1.shape
    pre_model = VGG19_pre_trained()
    epoch =25 
    batch_size = 50
    iteration = X0.shape[0] / batch_size
    print iteration
    epoch_loss = []
    epoch_accuracy = []
    print "iteration"
    for e in range(epoch):
        accuracy = 0
        total_len = 0
        loss =  0 
        
        # indices = np.arange(X0.shape[0])
        for i in range(iteration):        
            c = 0
            X0_batch = None
            X1_batch = None
            y_batch = None
            optimizer = optim.SGD(pre_model.classifier.parameters(), lr=0.001, momentum=0.9, nesterov=True,weight_decay = 1e-6)
            idx = np.random.choice(np.arange(X0.shape[0]), batch_size)
            #idx = indices[c:c + 50]
            X0_batch =X0[idx]
            X1_batch = X1[idx]
            y_batch = y[idx]
            fx0, fx1 = forward(X0_batch,X1_batch, pre_model)
            fx0_t = fx0.data.cpu().numpy()
            fx1_t = fx1.data.cpu().numpy()
            a= np.array([1 if x0 > x1 else -1 for (x0,x1) in zip(fx0_t, fx1_t)  ])
            model, running_loss = train_model(pre_model, fx0,fx1, y_batch, optimizer)
            accuracy += np.sum(a == y_batch)
            loss+= running_loss
            total_len += len(y_batch)
            c +=50
            print accuracy
        print float(accuracy)/float(X0.shape[0]) , "for epoch", e
        print float(loss)/float(X0.shape[0]), "loss"
        epoch_loss.append(float(loss)/float(X0.shape[0]))
        epoch_accuracy.append(float(accuracy)/float(X0.shape[0]))
    
    X0_val, X1_val , y_val =  validation
    X0_val = np.transpose(X0_val, (0, 3, 1,2))
    X1_val = np.transpose(X1_val, (0, 3, 1,2))
    iteration = X0_val.shape[0]/batch_size
    accuracy = 0
    #model.save_state_dict('mytraining.pt')
    thefile = open('test.txt', 'wr')
    for i in epoch_loss:
        print>>thefile, i
    acc_file = open("acc.text", "wr")
    for i in epoch_accuracy:
	print >> acc_file,i
    for i in range(iteration):
        c = 0
        fx0, fx1 = forward(X0_val[c:c+50], X1_val[c:c+50], pre_model)
        fx0_t = fx0.data.cpu().numpy()
        fx1_t = fx1.data.cpu().numpy()
        a= np.array([1 if x0 > x1 else -1 for (x0,x1) in zip(fx0_t, fx1_t)  ])
        accuracy += np.sum(a == y_val[c:c+50])
        # train_model(pre_x model, fx0,fx1, y_batch, optimizer)
        print np.sum(a==y_val[c:c:50])
        c += 50
    print float(accuracy)/float(X0_val.shape[0]), "val accuracy" 
    model.save_state_dict('mytraining.pt')

