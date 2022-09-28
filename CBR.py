import torch
import torch.nn as nn

class OneD_Conv_BN_ReLU(nn.Module):
    def __init__(self,in_channel_count,out_channel_count,kernel=3,stride=1,padding=0):
        super().__init__()
        
        self.seq = nn.Sequential(
          nn.Conv1d(in_channel_count,out_channel_count,kernel,stride,padding,padding_mode='replicate'),
          nn.BatchNorm1d(out_channel_count),
          nn.ReLU()
        )

    def forward(self, input):
        return self.seq.forward(input);


class Double_OneD_Conv_BN_ReLU(nn.Module):
    def __init__(self,in_channel_count,mid_channel_count,out_channel_count,kernel=3,stride=1,padding=0):
        super().__init__()

        self.seq = nn.Sequential(
          OneD_Conv_BN_ReLU(in_channel_count,mid_channel_count,kernel,stride,padding),
          OneD_Conv_BN_ReLU(mid_channel_count,out_channel_count,kernel,stride,padding)
        )

    def forward(self, input):
        return self.seq.forward(input);


class LSTM_Suit(nn.Module):
    def __init__(self,in_channel_count,out_channel_count,layer_count,batch_count):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lstm=nn.LSTM(input_size=in_channel_count,hidden_size=out_channel_count,num_layers=layer_count,batch_first=True)
        self.layer_count=layer_count
        self.batch_count=batch_count
        self.out_channel_count=out_channel_count

    def forward(self, input):
        d=torch.transpose(input,1,2)
        h0=torch.randn(self.layer_count, self.batch_count, self.out_channel_count).to(self.device)
        c0=torch.randn(self.layer_count, self.batch_count, self.out_channel_count).to(self.device)
        d,(h,c)=self.lstm.forward(d,(h0,c0))
        #self.h,self.c=h,c
        return torch.transpose(d,1,2)

