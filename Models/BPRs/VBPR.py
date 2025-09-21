import torch
from torch.nn.init import uniform_, normal_
from torch.nn import *
import torch.nn as nn
import torch.nn.functional as F

class VBPR(Module):
    def __init__(self, item_num, hidden_dim, visual_feature_dim, visual_features, with_Nor = True):
        super(VBPR, self).__init__()
        self.hidden_dim = hidden_dim
        self.with_Nor = with_Nor
        self.visual_features = visual_features
        self.item_num = item_num
        self.itemEmb = F.normalize(torch.normal(mean=torch.zeros(self.item_num + 1, self.hidden_dim), std=1/(self.hidden_dim)**0.5), p=2, dim=-1)
        self.itemB = torch.zeros([self.item_num + 1, 1])
        self.item_embs = nn.Embedding.from_pretrained(self.itemEmb, freeze=False, padding_idx=self.item_num)
        self.item_bias = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.item_bias_v = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.visual_nn_comp = Sequential(
            Linear(visual_feature_dim, self.hidden_dim),
            nn.Sigmoid())  # visual latent projection

        self.visual_nn_comp[0].apply(lambda module: normal_(module.weight.data,mean=0,std=1/(self.hidden_dim)**0.5))
        self.visual_nn_comp[0].apply(lambda module: normal_(module.bias.data,mean=0,std=1/(self.hidden_dim)**0.5))

    def forward(self, tops, bottoms): # for one single <top, bottom> sim score prediction.
        batchsize = len(tops)      
        tops_embs = self.item_embs(tops)              # get top ID embedding
        tops_bias = self.item_bias(tops)              # get top ID bias for similarity modeling 
        bottoms_embs = self.item_embs(bottoms)        # get bottom ID embedding
        bottoms_bias = self.item_bias(bottoms)        # get bottom ID bias for similarity modeling

        tops_v = self.visual_features[tops]           # get top visual feature from well trained ResNet
        bottoms_v = self.visual_features[bottoms]
        tops_latent_v = self.visual_nn_comp(tops_v)   # project top to visual latent space
        bottoms_latent_v = self.visual_nn_comp(bottoms_v)
        tops_bias_v = self.item_bias_v(tops)          #get top visaul similarity bias
        bottoms_bias_v = self.item_bias_v(bottoms)
        if self.with_Nor:                             # for eliminating the effect of different feature scale
            tops_embs = F.normalize(tops_embs, dim=0)
            tops_bias = F.normalize(tops_bias, dim=0)
            bottoms_embs = F.normalize(bottoms_embs, dim=0)
            bottoms_bias = F.normalize(bottoms_bias, dim=0)
            tops_latent_v = F.normalize(tops_latent_v, dim=0)
            bottoms_latent_v = F.normalize(bottoms_latent_v, dim=0)
            tops_bias_v = F.normalize(tops_bias_v, dim=0)
            bottoms_bias_v = F.normalize(bottoms_bias_v, dim=0)

        pred = tops_bias.view(batchsize) + bottoms_bias.view(batchsize) + F.cosine_similarity(tops_embs, bottoms_embs, dim=-1)
        pred += tops_bias_v.view(batchsize) + bottoms_bias_v.view(batchsize) + F.cosine_similarity(tops_latent_v, bottoms_latent_v, dim=-1)
        return pred # simply treat the ID embedding and feature similaity as the same importance, or flexible for adding hyperweight

    def bpr_loss(self, pos_score, neg_score): # BPR LOSS
        loss = - F.logsigmoid(pos_score - neg_score)
        loss = torch.mean(loss)
        return loss

    def fit(self, batch):  # for training 
        Us = batch[0] # user -> not useful
        Is = batch[1] # tops
        Js = batch[2] # pos bottoms
        Ks = batch[3] # neg bottoms

        pos_pred = self.forward(Is, Js)
        neg_pred = self.forward(Is, Ks)
        loss = self.bpr_loss(pos_pred, neg_pred)
        return loss

    def inference(self, batch): # for evaluation
        Us = batch[0] 
        Is = batch[1]
        Js = batch[2]
        Ks = batch[3]

        pos_pred = self.forward(Is, Js)
        neg_pred = self.forward(Is, Ks)
        return pos_pred - neg_pred # a good predicter should get a result > 0

    