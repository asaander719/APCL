
import torch
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_
from torch.nn import *
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from util.utils import get_parser
from Models.BPRs.BPR import BPR
from Models.BPRs.VTBPR import VTBPR
from Models.BPRs.TextCNN import TextCNN

class NiPCBPR(nn.Module):
    def __init__(self, args, embedding_weight, visual_features, text_features):        
        super(NiPCBPR, self).__init__()
        self.args = args
        self.weight_P = args.weight_P
        self.hidden_dim = args.hidden_dim
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.with_visual = args.with_visual
        self.with_text = args.with_text
        self.with_Nor = args.with_Nor
        self.cos = args.cos
        self.iPC = args.iPC

        self.visual_nn = nn.Sequential(
            nn.Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.visual_nn.apply(self.init_weights)

        self.p_visual_nn = nn.Sequential(
            nn.Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.p_visual_nn.apply(self.init_weights)

        self.iPC_visual_nn = nn.Sequential(
            nn.Linear(args.visual_feature_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        self.iPC_visual_nn.apply(self.init_weights)

        if args.dataset == 'IQON3000':
            self.text_nn = nn.Sequential(
                nn.Linear(100 * args.textcnn_layer, self.hidden_dim),
                nn.Sigmoid()
            )
        elif args.dataset == 'Polyvore':
            self.text_nn = nn.Sequential(
                nn.Linear(args.text_feature_dim, self.hidden_dim),
                nn.Sigmoid()
            )

        self.text_nn.apply(self.init_weights)
        self.p_text_nn = self.text_nn
        self.iPC_text_nn = self.text_nn

        if self.with_visual:
            self.visual_features = visual_features.cuda()
        if self.with_text:
            self.text_features = text_features.cuda()
            self.text_embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=False)
            self.textcnn = TextCNN(args.textcnn_layer, sentence_size=(args.max_sentence, args.text_feature_dim), output_size=self.hidden_dim)

        self.vtbpr = VTBPR(self.user_num, self.item_num, hidden_dim=self.hidden_dim, 
                           theta_text=self.with_text, theta_visual=self.with_visual, with_Nor=True, cos=True)
        print(f'Module already prepared, {self.user_num} users, {self.item_num} items')
        self.bpr = BPR(self.user_num, self.item_num)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.uniform_(module.weight.data, 0, 0.001)
            nn.init.uniform_(module.bias.data, 0, 0.001)

    def forward(self, batch, train, **kwargs):
        Us, Is, Js, Ks, bhis, this, tbhis = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        bs = len(Us)

        if self.with_visual:
            vis_I, vis_J, vis_K = self.visual_features[Is], self.visual_features[Js], self.visual_features[Ks]
  
            I_visual_latent, J_visual_latent, K_visual_latent = self.visual_nn(vis_I), self.visual_nn(vis_J), self.visual_nn(vis_K)
            J_visual_latent_p, K_visual_latent_p = self.p_visual_nn(vis_J), self.p_visual_nn(vis_K)

            if self.with_Nor:
                I_visual_latent, J_visual_latent, K_visual_latent = F.normalize(I_visual_latent, dim=0), F.normalize(J_visual_latent, dim=0), F.normalize(K_visual_latent, dim=0)
                J_visual_latent_p, K_visual_latent_p = F.normalize(J_visual_latent_p, dim=0), F.normalize(K_visual_latent_p, dim=0)

            if self.cos:
                visual_ij, visual_ik = F.cosine_similarity(I_visual_latent, J_visual_latent, dim=-1), F.cosine_similarity(I_visual_latent, K_visual_latent, dim=-1)
            else:
                visual_ij, visual_ik = torch.sum(I_visual_latent * J_visual_latent, dim=-1), torch.sum(I_visual_latent * K_visual_latent, dim=-1)

            if self.iPC:    
                vis_this = self.visual_features[this]
                vis_this = self.iPC_visual_nn(vis_this)
                vis_J_c, vis_K_c = self.iPC_visual_nn(vis_J), self.iPC_visual_nn(vis_K)
                t_his_visual = torch.mean(vis_this, dim=-2)

                if self.with_Nor:
                    t_his_visual, vis_J_c, vis_K_c = F.normalize(t_his_visual, dim=0), F.normalize(vis_J_c, dim=0), F.normalize(vis_K_c, dim=0)

                if self.cos:
                    Visual_TuJ, Visual_TuK = F.cosine_similarity(t_his_visual, vis_J_c, dim=-1), F.cosine_similarity(t_his_visual, vis_K_c, dim=-1)
                else:
                    Visual_TuJ, Visual_TuK = torch.sum(t_his_visual * vis_J_c, dim=-1), torch.sum(t_his_visual * vis_K_c, dim=-1)

        if self.with_text:
            text_I, text_J, text_K = self.text_features[Is], self.text_features[Js], self.text_features[Ks]

            if self.args.dataset == 'IQON3000':
                text_I, text_J, text_K = self.text_embedding(text_I), self.text_embedding(text_J), self.text_embedding(text_K)
                I_text_fea, J_text_fea, K_text_fea = self.textcnn(text_I.unsqueeze(1)), self.textcnn(text_J.unsqueeze(1)), self.textcnn(text_K.unsqueeze(1))
            elif self.args.dataset == 'Polyvore':
                I_text_fea, J_text_fea, K_text_fea = self.text_nn(text_I), self.text_nn(text_J), self.text_nn(text_K)

            I_text_latent, J_text_latent, K_text_latent = self.text_nn(I_text_fea), self.text_nn(J_text_fea), self.text_nn(K_text_fea)
            J_text_latent_p, K_text_latent_p = self.p_text_nn(J_text_fea), self.p_text_nn(K_text_fea)

            if self.with_Nor:
                I_text_latent, J_text_latent, K_text_latent = F.normalize(I_text_latent, dim=0), F.normalize(J_text_latent, dim=0), F.normalize(K_text_latent, dim=0)
                J_text_latent_p, K_text_latent_p = F.normalize(J_text_latent_p, dim=0), F.normalize(K_text_latent_p, dim=0)

            if self.cos:
                text_ij, text_ik = F.cosine_similarity(I_text_latent, J_text_latent, dim=-1), F.cosine_similarity(I_text_latent, K_text_latent, dim=-1)
            else:
                text_ij, text_ik = torch.sum(I_text_latent * J_text_latent, dim=-1), torch.sum(I_text_latent * K_text_latent, dim=-1)
                
            if self.iPC:
                if self.args.dataset == 'IQON3000':
                    text_this = self.text_embedding(self.text_features[this])
                    this_text_fea = self.textcnn(text_this.reshape(bs * self.args.num_his, self.args.max_sentence, self.args.text_feature_dim).unsqueeze(1))
                    this_text_fea = self.iPC_text_nn(this_text_fea).reshape(bs, self.args.num_his, self.hidden_dim)
                    this_text_fea_mean = torch.mean(this_text_fea, dim=-2)

                    text_J_c, text_K_c = self.iPC_text_nn(J_text_fea), self.iPC_text_nn(K_text_fea)
                    if self.with_Nor:
                        this_text_fea_mean, text_J_c, text_K_c = F.normalize(this_text_fea_mean, dim=0), F.normalize(text_J_c, dim=0), F.normalize(text_K_c, dim=0)

                    if self.cos:
                        text_TuJ, text_TuK = F.cosine_similarity(this_text_fea_mean, text_J_c, dim=-1), F.cosine_similarity(this_text_fea_mean, text_K_c, dim=-1)
                    else:
                        text_TuJ, text_TuK = torch.sum(this_text_fea_mean * text_J_c, dim=-1), torch.sum(this_text_fea_mean * text_K_c, dim=-1)

                elif self.args.dataset == 'Polyvore':
                    text_this = self.text_features[this]
                    text_this = self.iPC_text_nn(text_this)
                    text_J_c, text_K_c = self.iPC_text_nn(text_J), self.iPC_text_nn(text_K)
                    t_his_text = torch.mean(text_this, dim=-2)
                    if self.with_Nor:
                        t_his_text, text_J_c, text_K_c = F.normalize(t_his_text, dim=0), F.normalize(text_J_c, dim=0), F.normalize(text_K_c, dim=0)

                    if self.cos:
                        text_TuJ, text_TuK = F.cosine_similarity(t_his_text, text_J_c, dim=-1), F.cosine_similarity(t_his_text, text_K_c, dim=-1)
                    else:
                        text_TuJ, text_TuK = torch.sum(t_his_text * text_J_c, dim=-1), torch.sum(t_his_text * text_K_c, dim=-1) 

        if self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, J_text_latent_p)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, J_text_latent)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, K_text_latent)

            p_ij, p_ik = 0.5 * (visual_ij + text_ij), 0.5 * (visual_ik + text_ik)

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk) 

            if self.iPC:
                C_TuJ = self.args.iPC_w * (self.args.iPC_v_w * Visual_TuJ + (1-self.args.iPC_v_w) * text_TuJ)
                C_TuK = self.args.iPC_w * (self.args.iPC_v_w * Visual_TuK + (1-self.args.iPC_v_w) * text_TuK)
                pred = pred + C_TuJ - C_TuK  

        if self.with_visual and not self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, J_visual_latent_p, None)
                cuk = self.vtbpr(Us, Ks, K_visual_latent_p, None)
            else:
                cuj = self.vtbpr(Us, Js, J_visual_latent, None)
                cuk = self.vtbpr(Us, Ks, K_visual_latent, None)

            p_ij, p_ik = visual_ij, visual_ik 

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.iPC:
                C_TuJ = self.args.iPC_w * Visual_TuJ 
                C_TuK = self.args.iPC_w * Visual_TuK 

                pred = pred + C_TuJ - C_TuK 

        if not self.with_visual and self.with_text:
            if self.args.b_PC:
                cuj = self.vtbpr(Us, Js, None, J_text_latent_p)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent_p)
            else:
                cuj = self.vtbpr(Us, Js, None, J_text_latent)
                cuk = self.vtbpr(Us, Ks, None, K_text_latent)

            p_ij, p_ik = 0.5 * text_ij, 0.5 * text_ik

            pred = self.weight_P * p_ij + (1 - self.weight_P) * cuj - (self.weight_P * p_ik + (1 - self.weight_P) * cuk)

            if self.iPC:
                C_TuJ = self.args.iPC_w * text_TuJ
                C_TuK = self.args.iPC_w * text_TuK

                pred = pred + C_TuJ - C_TuK 

        return pred


               

