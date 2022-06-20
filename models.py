import torch.nn as nn
import train
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

class AAAA(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        #self.linear_1 = nn.Linear(4*input_size, hidden_layer_size)
        #self.relu = nn.ReLU()
        #self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(60, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        print(batchsize)

        # layer 1
        #x = self.linear_1(x)
        #x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        print('ss')
        print(h_n.shape)
        print(lstm_out)
        print(c_n)
        
        # Linear layer로 보내기 위해서 output의 shape을 바꿔준다.
        x = h_n.permute(1, 0, 2).reshape(4, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

class LSTMModel(nn.Module):
    def __init__(self, input_dim = 1 , hidden_dim = 200, num_layers =2, output_dim = 1):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (z, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        #print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x.view(-1,100,1), (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]).squeeze()
        # out.size() --> 100, 10
        return out
    
class LSTMModelSS(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=2):
      super().__init__()
      self.hidden_layer_size = hidden_layer_size
      self.num_layers = num_layers
      self.hidden_dim = hidden_layer_size

      self.lstm = nn.LSTM(input_size, hidden_layer_size,num_layers=2, batch_first = True)

      self.linear = nn.Linear(hidden_layer_size, output_size)

      #self.hidden_cell = (torch.zeros(num_layers,4,self.hidden_layer_size),
      #                    torch.zeros(num_layers,4,self.hidden_layer_size)).to(device)

  def forward(self, x):
      #print(input_seq.view(4,60, 1))
      #print(input_seq)
      #print(self.hidden_cell)
      #print(torch.zeros(1,1,self.hidden_layer_size).shape)
      h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).requires_grad_()
      c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).requires_grad_()
      lstm_out, self.hidden_cell = self.lstm(x.view(4,60,1), (h0.detach(),c0.detach()))
      print(lstm_out.type)
      print(lstm_out.shape)
      predictions = self.linear(lstm_out[:,-1,:])
      return predictions