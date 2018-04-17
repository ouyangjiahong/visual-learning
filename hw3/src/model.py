import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    '''
    define the CNN model as a classifier
    '''
    def __init__(self, num_class=51):
        super(Classifier, self).__init__()
        self.num_class=num_class
        self.build_classifier()

    def build_classifier(self):
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Linear(in_features=512, out_features=self.num_class, bias=True))
        # for layer in self.classifier:
        #     nn.init.normal(layer.weight)

    def forward(self, x):
        x = self.classifier(x)  # (bs, 512) -> (bs, 51)
        x = torch.mean(x, dim=0, keepdim=True)    # (51,)
        return x

class RNN(nn.Module):
    '''
    define the CNN model as a classifier
    '''
    def __init__(self, num_class=51, hidden_size=256, time_step=10):
        super(RNN, self).__init__()
        self.num_class=num_class
        self.hidden_size = hidden_size
        self.time_step = time_step
        self.build_rnn()

    def build_rnn(self):
        self.rnn = nn.RNN(input_size=512, hidden_size=self.hidden_size, num_layers=2,
                        batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(in_features=self.hidden_size, out_features=self.num_class, bias=True)

    def forward(self, x):
        bs = x.size()[0]
        output, _ = self.rnn(x)         # (bs, 10, 512) -> (bs, 10, 512)
        x = output[:,-1,:]              # (bs, 512)
        x = self.classifier(x.view(bs, self.hidden_size))  # (bs, 512) -> (bs, 51)
        return x

class RNN_simple(nn.Module):
    '''
    define the CNN model as a classifier
    '''
    def __init__(self, num_class=51, time_step=10):
        super(RNN, self).__init__()
        self.num_class=num_class
        self.time_step = time_step
        self.build_rnn()

    def build_rnn(self):
        self.rnn = nn.RNN(input_size=512, hidden_size=self.num_class, num_layers=1,
                        batch_first=True)

    def forward(self, x):
        bs = x.size()[0]
        output, _ = self.rnn(x)         # (bs, 10, 512) -> (bs, 10, 512)
        x = output[:,-1,:]              # (bs, 512)
        return x
