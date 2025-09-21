# import torch
# from torch.nn.init import uniform_
# from torch.nn import *
# import torch.nn as nn

# class TextCNN(Module):
#     def __init__(self, layer, sentence_size = (83, 300), output_size = 512, uniform=False):
#         super(TextCNN, self).__init__()
#         self.max_sentense_length, self.word_vector_size = sentence_size
#         self.text_cnn_layers = [
#             nn.Sequential(
#               nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(2,self.word_vector_size),stride=1),
#               nn.Sigmoid(),
#               nn.MaxPool2d(kernel_size=(self.max_sentense_length - 1,1),stride=1))]
#         if layer > 1:
#             self.text_cnn_layers.append(
#             nn.Sequential(
#               nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(3,self.word_vector_size),stride=1),
#               nn.Sigmoid(),
#               nn.MaxPool2d(kernel_size=(self.max_sentense_length - 2,1),stride=1)))
#         if layer > 2:
#             self.text_cnn_layers.append(
#             nn.Sequential(
#               nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(4,self.word_vector_size),stride=1),
#               nn.Sigmoid(),
#               nn.MaxPool2d(kernel_size=(self.max_sentense_length - 3,1),stride=1)))
#         if layer > 3:
#             self.text_cnn_layers.append(
#             nn.Sequential(
#               nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(5,self.word_vector_size),stride=1),
#               nn.Sigmoid(),
#               nn.MaxPool2d(kernel_size=(self.max_sentense_length - 4,1),stride=1)))
            
#         self.text_cnn = ModuleList(self.text_cnn_layers)
        
#         if uniform == True:
#             for i in range(layer):
#                 init.uniform_(self.text_cnn[i][0].weight.data, 0, 0.001)
#                 init.uniform_(self.text_cnn[i][0].bias.data, 0, 0.001)
                    
#     def forward(self, input):
#         cnn_features = [nn.Conv2d(input).squeeze_(-1).squeeze_(-1) for nn.Conv2d in self.text_cnn]
#         cnn_features = torch.cat(cnn_features, 1)
#         #return self.text_nn(cnn_features)
#         return cnn_features

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, num_layers, sentence_size=(83, 300), output_size=512, uniform=False):
        super(TextCNN, self).__init__()
        self.max_sentence_length, self.word_vector_size = sentence_size
        self.text_cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 100, kernel_size=(i, self.word_vector_size), stride=1),
                nn.Sigmoid(),
                nn.MaxPool2d(kernel_size=(self.max_sentence_length - i + 1, 1), stride=1)
            ) for i in range(2, num_layers + 2)
        ])

        if uniform:
            for layer in self.text_cnn_layers:
                nn.init.uniform_(layer[0].weight.data, 0, 0.001)
                nn.init.uniform_(layer[0].bias.data, 0, 0.001)

    def forward(self, x):
        cnn_features = [layer(x).squeeze_(-1).squeeze_(-1) for layer in self.text_cnn_layers]
        cnn_features = torch.cat(cnn_features, 1)
        return cnn_features

