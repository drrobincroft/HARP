import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import CBR as cbr
import PrepareData as pd
import numpy as np
import torchsummary as ts

class STGDiscriminator_Harp(nn.Module):
    def __init__(self):
        super().__init__()

        self.ext=nn.Sequential(
            cbr.Double_OneD_Conv_BN_ReLU(26+104,256,256),
            nn.MaxPool1d(2,stride=2),
            cbr.Double_OneD_Conv_BN_ReLU(256,512,512),
            nn.MaxPool1d(2,stride=2),
            cbr.Double_OneD_Conv_BN_ReLU(512,1024,1024)
            )

        self.mlp=nn.Sequential(
            nn.Linear(1024,512),
            nn.Linear(512,1),
            nn.Sigmoid()
            )

    def forward(self, input):
        feats=self.ext.forward(input)
        feats=torch.transpose(feats,1,2)
        res=self.mlp.forward(feats)
        return torch.mean(res,1)


def STGDiscriminatorLoss(output, target):
    return torch.mean(torch.log(torch.abs(output-target)+1.0e-8))


def train_in_one_epoch(input, target, model, optimizer, scheduler, minibatchsize=6):
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])

    model.train()  # Set model to training mode    
    
    epoch_samples = 0
    sumloss= 0.0
    for k in range(0, input.size(0)-minibatchsize, minibatchsize):
        
        optimizer.zero_grad() # zero the parameter gradients    
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            output = model(input[k:k+minibatchsize])
            loss = STGDiscriminatorLoss(output, target[k:k+minibatchsize])

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # statistics
            epoch_samples += minibatchsize
            sumloss+=loss.data.cpu().numpy()*minibatchsize

    scheduler.step()
    epoch_loss = sumloss/epoch_samples
    print("Training Loss: {}".format(epoch_loss))
    return epoch_loss
    

def validate_in_one_epoch(validinput, validtarget, model, optimizer, scheduler, minibatchsize):
    model.eval()   # Set model to evaluate mode

    epoch_samples = 0
    sumloss= 0.0
    accurancy=0.0
    for k in range(0, validinput.size(0)-minibatchsize, minibatchsize):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            output = model(validinput[k:k+minibatchsize])
            loss = STGDiscriminatorLoss(output, validtarget[k:k+minibatchsize])

            # statistics
            epoch_samples += minibatchsize
            sumloss+=loss.data.cpu().numpy()*minibatchsize
            accurancy+=torch.sum(torch.abs(torch.round(output)-validtarget[k:k+minibatchsize]))

    epoch_loss = sumloss/epoch_samples
    print("Validating Loss: {}, Accurancy: {}".format(epoch_loss,1.0-accurancy/epoch_samples))
    return epoch_loss

def train_model(input, target, validinput, validtarget, num_epochs=25):
    #prepare the misc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = STGDiscriminator_Harp().to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    ts.summary(model,(input.size(1),input.size(2)))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    minibatchsize=25

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        train_in_one_epoch(input.to(device), target.to(device), model, optimizer_ft, exp_lr_scheduler, minibatchsize)
        epoch_loss = validate_in_one_epoch(validinput.to(device), validtarget.to(device), model, optimizer_ft, exp_lr_scheduler, minibatchsize)

        # deep copy the model
        if epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def PrepareTrainingAndValidatingData():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gen=torch.load("gen_harp.pth").to(device)

    train_wav_data, train_body_data, validate_wav_data, validate_body_data=pd.ReadTrainDataAndValidData_Harp()
    minibatch=40

    #product false sample for training
    batch_num=round(train_wav_data.size(0)*0.5)
    len=train_wav_data.size(2)//8*8 #it must be 16 times, or the length of input and output may not match
    inds=torch.randperm(batch_num)
    X=train_wav_data[inds,:,0:len]
    #Y=torch.cat((X,torch.zeros(batch_num,6,len)),dim=1)
    Y=X
    F=torch.tensor([])
    with torch.set_grad_enabled(False):
        for k in range(0,Y.size(0)-minibatch,minibatch):
            print(k,Y.size(0))
            F=torch.cat((F,torch.cat((X[k:k+minibatch], gen(Y[k:k+minibatch].to(device)).to("cpu")),1)),0)
    print(F.shape)
    #product true sample for training
    inds=torch.randperm(batch_num)
    T=torch.cat((train_wav_data[inds,:,0:len],train_body_data[inds,:,0:len]),1)
    print(T.shape)
    inds=torch.randperm(F.size(0)+T.size(0))
    traininginput=torch.cat((F,T),0)[inds]
    trainingtarget=torch.cat((torch.zeros(F.size(0),1),torch.ones(T.size(0),1)),0)[inds]

    batch_num=round(validate_wav_data.size(0)*0.3)
    len=validate_wav_data.size(2)//8*8
    inds=torch.randperm(batch_num)
    X=validate_wav_data[inds,:,0:len]
    #product false sample for validating
    #Y=torch.cat((X,torch.zeros(batch_num,6,len)),dim=1)
    Y=X
    F=torch.tensor([])
    with torch.set_grad_enabled(False):
        for k in range(0,Y.size(0)-minibatch,minibatch):
            print(k,Y.size(0))
            F=torch.cat((F,torch.cat((X[k:k+minibatch], gen(Y[k:k+minibatch].to(device)).to("cpu")),1)),0)
    #produce true sample for validating
    inds=torch.randperm(batch_num)
    T=torch.cat((validate_wav_data[inds,:,0:len],validate_body_data[inds,:,0:len]),1)
    inds=torch.randperm(F.size(0)+T.size(0))
    validatinginput=torch.cat((F,T),0)[inds]
    validatingtarget=torch.cat((torch.zeros(F.size(0),1),torch.ones(T.size(0),1)),0)[inds]
    
    return traininginput, trainingtarget, validatinginput, validatingtarget

def Train(num_epochs=3):
    traininginput, trainingtarget, validatinginput, validatingtarget=PrepareTrainingAndValidatingData()
    #np.save("xx.npy",traininginput.data.numpy())
    print(traininginput.shape,trainingtarget.shape,validatinginput.shape, validatingtarget.shape)
    disc = train_model(traininginput, trainingtarget, validatinginput, validatingtarget, num_epochs)
    torch.save(disc,"disc_harp.pth")

def PrepareTestData():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gen=torch.load("gen_harp.pth").to(device)

    train_wav_data, train_body_data=pd.ReadTestData_Harp()
    minibatch=200

    #product false sample for training
    batch_num=round(train_wav_data.size(0)*0.5)
    len=train_wav_data.size(2)//8*8
    inds=torch.randperm(batch_num)
    X=train_wav_data[inds,:,0:len]
    #Y=torch.cat((X,torch.zeros(batch_num,6,len)),dim=1)
    Y=X
    if Y.size(0)>minibatch:
        F=torch.tensor([])
        with torch.set_grad_enabled(False):
            for k in range(0,Y.size(0)-minibatch,minibatch):
                print(k,Y.size(0))
                F=torch.cat((F,torch.cat((X[k:k+minibatch], gen(Y[k:k+minibatch].to(device)).to("cpu")),1)),0)
    else:
        F=torch.cat((X, gen(Y.to(device)).to("cpu")),1)
    print(F.shape)
    #product true sample for training
    inds=torch.randperm(batch_num)
    T=torch.cat((train_wav_data[inds,:,0:len],train_body_data[inds,:,0:len]),1)
    print(T.shape)
    inds=torch.randperm(F.size(0)+T.size(0))
    traininginput=torch.cat((F,T),0)[inds]
    trainingtarget=torch.cat((torch.zeros(F.size(0),1),torch.ones(T.size(0),1)),0)[inds]

    return traininginput, trainingtarget

def Test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    disc=torch.load("disc_harp.pth")
    disc=disc.to(device)
    ts.summary(disc,(26+104,425))
    testinput, testtarget=PrepareTestData()
    print(testinput.shape, testtarget.shape)
    
    output=disc(testinput.to(device))
    print("Test accurancy: {}".format(1.0-torch.sum(torch.abs(torch.round(output)-testtarget.to(device)))/testinput.size(0)))
