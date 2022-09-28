#Harp gesture generator

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import CBR as cbr
import PrepareData as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt

class Harp(nn.Module):
    def __init__(self,input_channel_num,output_channel_num,minibatchsize):
        super().__init__()

        self.encoder_01 = cbr.Double_OneD_Conv_BN_ReLU(input_channel_num,64,64,3,1,1)

        self.encoder_02=nn.Sequential(
            nn.MaxPool1d(2,stride=2),
            cbr.Double_OneD_Conv_BN_ReLU(64,128,128,3,1,1)
            )

        self.encoder_03=nn.Sequential(
            nn.MaxPool1d(2,stride=2),
            cbr.Double_OneD_Conv_BN_ReLU(128,256,256,3,1,1)
            )

        self.encoder_04=nn.Sequential(
            nn.MaxPool1d(2,stride=2),
            cbr.Double_OneD_Conv_BN_ReLU(256,512,512,3,1,1)
            )

        self.decoder_04=nn.Sequential(
            cbr.LSTM_Suit(512,512,1,minibatchsize),
            nn.Upsample(scale_factor=2.0)
            )        

        self.decoder_03=nn.Sequential(
            cbr.OneD_Conv_BN_ReLU(256+512,512,3,1,1),
            cbr.LSTM_Suit(512,512,1,minibatchsize),
            nn.Upsample(scale_factor=2.0)
            )

        self.decoder_02=nn.Sequential(
            cbr.OneD_Conv_BN_ReLU(128+512,256,3,1,1),
            cbr.LSTM_Suit(256,256,1,minibatchsize),
            nn.Upsample(scale_factor=2.0)
            )

        self.decoder_01=nn.Sequential(
            cbr.OneD_Conv_BN_ReLU(64+256,256,3,1,1),
            cbr.LSTM_Suit(256,256,1,minibatchsize),
            nn.Conv1d(256,256,1,1,0),
            cbr.LSTM_Suit(256,output_channel_num,1,minibatchsize)
            )
        
    def forward(self, input):
        x1=self.encoder_01.forward(input)
        x2=self.encoder_02.forward(x1)
        x3=self.encoder_03.forward(x2)
        x4=self.encoder_04.forward(x3)

        y3=self.decoder_04.forward(x4)
        y3=torch.cat((y3,x3),dim=1)
        y2=self.decoder_03.forward(y3)
        y2=torch.cat((y2,x2),dim=1)
        y1=self.decoder_02.forward(y2)
        y1=torch.cat((y1,x1),dim=1)
        gesture=self.decoder_01.forward(y1)

        return gesture

def STGL1Loss(generated, target, thredhold):
    #return torch.mean(F.relu(torch.abs(generated-target)-thredhold))
    return torch.sum(torch.abs(generated-target))

def STGDiscriminatorLoss2(generated,target,mfcc,disc):
    #with torch.set_grad_enabled(False):
    return torch.mean(torch.abs(1.0-disc(torch.cat((mfcc,generated),dim=1))+1.0e-8))+torch.mean(torch.abs(disc(torch.cat((mfcc,target),dim=1))))
    #return torch.mean(torch.abs(1.0-disc(torch.cat((mfcc,generated),dim=1))))

def STGGeneratorLoss(generated, target, disc, mfcc, weights, thredhold):
    return weights[0]*STGL1Loss(generated, target, thredhold)+weights[1]*STGDiscriminatorLoss2(generated,target,mfcc,disc)

def PrepareTrainingAndValidData():
    train_wav_data, train_body_data, validate_wav_data, validate_body_data=pd.ReadTrainDataAndValidData_Harp()

    batch_num=round(train_wav_data.size(0)*0.9)
    inds=torch.randperm(batch_num)
    len=train_wav_data.size(2)//8*8  #it must be 8 times, or the length of input and output may not match
    traininginput=train_wav_data[inds,:,0:len];
    trainingtarget=train_body_data[inds,:,0:len];

    batch_num=round(min(validate_wav_data.size(0)*0.9,train_wav_data.size(0)*0.1))
    inds=torch.randperm(batch_num)
    len=validate_wav_data.size(2)//8*8
    validatinginput=validate_wav_data[inds,:,0:len];
    validatingtarget=validate_body_data[inds,:,0:len];

    return traininginput, trainingtarget, validatinginput, validatingtarget

def PrepareTestData():
    test_wav_data, test_body_data=pd.ReadTestData_Harp()

    batch_num=test_wav_data.size(0)
    len=test_wav_data.size(2)//8*8
    testinginput=test_wav_data[:,:,0:len];
    testingtarget=test_body_data[:,:,0:len]

    return testinginput, testingtarget

def train_in_one_epoch(input, target, model, optimizer, scheduler, disc, weights, thredhold, minibatchsize=6):
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])

    model.train()  # Set model to training mode    
    
    epoch_samples = 0
    sumloss= 0.0
    Loss1=0.0
    for k in range(0, input.size(0)-minibatchsize, minibatchsize):
        
        optimizer.zero_grad() # zero the parameter gradients    
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            output = model(input[k:k+minibatchsize])
            loss = STGGeneratorLoss(output, target[k:k+minibatchsize], disc, input[k:k+minibatchsize,0:26], weights, thredhold)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # statistics
            epoch_samples += minibatchsize
            sumloss+=loss.data.cpu().numpy()*minibatchsize
            Loss1+=STGL1Loss(output,target[k:k+minibatchsize],thredhold)*minibatchsize

    scheduler.step()
    epoch_loss = sumloss/epoch_samples
    Loss1/=epoch_samples
    print("Training Loss: {}, L1Loss: {}, Mean L1Loss: {}, ".format(epoch_loss,Loss1,Loss1/minibatchsize/target.size(1)/target.size(2)))
    return epoch_loss

def validate_in_one_epoch(validinput, validtarget, model, optimizer, scheduler, disc, weights, thredhold,dovalidatewithdiscriminator=True, minibatchsize=6):
    model.eval()   # Set model to evaluate mode

    epoch_samples = 0
    sumloss= 0.0
    Loss1=0.0
    Loss2=0.0
    for k in range(0, validinput.size(0)-minibatchsize, minibatchsize):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            output = model(validinput[k:k+minibatchsize])
            loss = STGGeneratorLoss(output, validtarget[k:k+minibatchsize], disc, validinput[k:k+minibatchsize,0:26], weights, thredhold)

            # statistics
            epoch_samples += minibatchsize
            sumloss+=loss.data.cpu().numpy()*minibatchsize
            Loss1+=STGL1Loss(output,validtarget[k:k+minibatchsize],thredhold)*minibatchsize
            if dovalidatewithdiscriminator:
                Loss2+=STGDiscriminatorLoss2(output,validtarget[k:k+minibatchsize],validinput[k:k+minibatchsize,0:26],disc)*minibatchsize
            else:
                Loss2+=0.0;

    epoch_loss = sumloss/epoch_samples
    Loss1/=epoch_samples
    Loss2/=epoch_samples
    print("Validating Loss: {}, L1Loss: {}, Mean L1Loss: {}, Fooling Discriminator Loss: {}".format(epoch_loss,Loss1,Loss1/minibatchsize/validtarget.size(1)/validtarget.size(2),Loss2))
    return epoch_loss

def train_model(input, target, validinput, validtarget, discfile, genfile, docreatenewmodel=False, dovalidatewithdiscriminator=True, num_epochs=25, learning_rate=5e-8):
    #prepare the misc
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)
    minibatchsize=40
    if docreatenewmodel:
        model=Harp(26,104,minibatchsize).to(device)
    else:
        model=torch.load(genfile).to(device)
    #ts.summary(gen,(32,424))
    print(model)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, [30000,35000], gamma=0.5)
    disc=torch.load(discfile).to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    weights=[1,10000];
    thredhold=0.2
        
    plt.ion()
    train_loss=np.array([])
    epoch_loss=np.array([])
    torch.backends.cudnn.enabled = False
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        train_loss=np.append(train_loss, train_in_one_epoch(input.to(device), target.to(device), model, optimizer_ft, exp_lr_scheduler, disc, weights,
                          thredhold, minibatchsize))
        epoch_loss=np.append(epoch_loss, validate_in_one_epoch(input.to(device), target.to(device), model, optimizer_ft, exp_lr_scheduler, disc, weights,
                          thredhold, dovalidatewithdiscriminator, minibatchsize))

        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(np.arange(0, epoch+1, 1), train_loss, '-b')
        plt.subplot(1,2,2)
        plt.plot(np.arange(0, epoch+1, 1), epoch_loss, '-r')
        plt.pause(0.15)

        # deep copy the model
        if epoch_loss[-1] < best_loss:
            print("saving best model")
            best_loss = epoch_loss[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch%10==9:
            print("saving final model")
            torch.save(model, genfile)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("saving final model")
    torch.save(model, genfile)

    print('Best val loss: {:4f}'.format(best_loss))
    plt.savefig("training loss curve.png",dpi=300)
    plt.ioff()
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def FirstTrain(num_epochs=3, learning_rate=1e-7):
    traininginput, trainingtarget, validatinginput, validatingtarget=PrepareTrainingAndValidData()
    print(traininginput.shape,trainingtarget.shape,validatinginput.shape, validatingtarget.shape)
    gen = train_model(traininginput, trainingtarget, validatinginput, validatingtarget, "disc_harp.pth", "gen_harp.pth", True, False, num_epochs, learning_rate)
    torch.save(gen,"gen_harp_best.pth")

def Train(num_epochs=3, learning_rate=1e-7):
    traininginput, trainingtarget, validatinginput, validatingtarget=PrepareTrainingAndValidData()
    print(traininginput.shape,trainingtarget.shape,validatinginput.shape, validatingtarget.shape)
    gen=train_model(traininginput, trainingtarget, validatinginput, validatingtarget, "disc_harp.pth", "gen_harp.pth", False, True, num_epochs, learning_rate)
    torch.save(gen,"gen_harp_best.pth")

def Test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    gen=torch.load("gen_harp.pth")
    gen=gen.to(device)
    print(gen)
    disc=torch.load("disc_harp.pth").to(device)
    weights=[1,10000];
    thredhold=0.2

    testinput, testtarget=PrepareTestData()
    print(testinput.shape, testtarget.shape)
    testinput=testinput.to(device)
    testtarget=testtarget.to(device)
    re=testinput.shape[0]%40   #it must be times of 40
    if re!=0:
        testinput2=torch.cat((testinput,torch.zeros((40-re,testinput.shape[1],testinput.shape[2])).to(device)),0)
        testtarget2=torch.cat((testtarget,torch.zeros((40-re,testtarget.shape[1],testtarget.shape[2])).to(device)),0)
    
    output2=gen(testinput2)
    print(output2.shape,re)
    output=output2[0:output2.shape[0]+re-40];
    print(output.shape)
    ert=STGL1Loss(output,testtarget,thredhold)
    print("Test Loss: {}, L1Loss: {}, L1Loss Mean: {}, Fooling Discriminator Loss: {}".format(STGGeneratorLoss(output,testtarget,disc,testinput, weights,thredhold),
                                                                             ert, ert/torch.numel(output),
                                                                             STGDiscriminatorLoss2(output,testtarget,testinput,disc)))

    np.save("generated.npy",output.data.cpu().numpy())