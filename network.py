import math
import torch
import torch.nn as nn 
import pandas as pd
import numpy as np

device = 'cpu'

class Mask(nn.Module):
    """
    Mask 분포
    """
    def __init__(self, dim):
        super().__init__()        
        self.sigma = 3.0
        self.noise = torch.randn(dim).to(device)
        self.mu = torch.tensor([5.0] * dim)
        self.mu = nn.Parameter(self.mu)

    def sample(self, noisy=True):
        noise = self.sigma * self.noise.normal_()
        mask = self.mu + noise*noisy
        mask = torch.softmax(mask, dim=0) 
        mask = torch.clamp(mask, 0.0, 1.0)
        return mask

class Rnet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

        self.BN1 = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm1d(128)
        self.act = nn.ReLU()

    def forward(self, w):
        x = self.layer1(w)
        x = self.BN1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.BN2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x
    

class Policy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.n_cut = 6
        self.dim = dim
        self.params = self.init()
    
    def init(self, min=1.0, max=2.0):
        """ 
        Cutting point initialize
        """
        params = nn.ParameterList()

        for _ in range(self.dim):
            step = (max-min) / self.n_cut
            cut_points_by_feature = [min + step * i for i in range(self.n_cut)]
            cut_points_by_feature = torch.tensor(cut_points_by_feature)
            cut_points_by_feature.requires_grad = True
            params.append(cut_points_by_feature)

        return params

    def forward(self, state):
        """
        Input: state (Batch X 종목 X 팩터값)  
        Output: action (Batch X 종목 X 각 팩터 점수)
        """
        action = []

        for i in range(self.dim):
            feature = state[:,:, i:i+1]
            points = self.params[i]
            points, _ = torch.sort(points)

            W = torch.arange(len(points) + 1) + 1
            b = torch.cumsum(-points, 0)
            b = torch.cat((torch.zeros(1), b))

            temp = 0.01
            logit = W * feature + b
            logit = torch.clamp(logit / temp, max=88)
            exponential = torch.exp(logit)
            summation = torch.sum(exponential, axis=-1, keepdim=True)
            prob = exponential / summation

            interval_num = self.n_cut + 1
            score = torch.tensor(range(1, interval_num+1)) * prob 
            score = torch.sum(score, axis=-1, keepdim=True)
            action.append(score)

        action = torch.cat(action, axis=-1)
        return action          


class Qnet(nn.Module):
    """
    DDPG의 Q network
    """
    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        
        # For state embedding 
        self.s_layer1 = nn.Linear(dim, 256)
        self.s_layer2 = nn.Linear(256, 1)

        # For action embedding
        self.a_layer1 = nn.Conv1d(1, 128, 30, 30)
        self.a_layer2 = nn.Conv1d(128, 64, 10, 1)

        # For q approximation
        self.layer1 = nn.Linear(300 + 64, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(128)
        self.BN2 = nn.BatchNorm1d(64)
                
    def forward(self, s, a):
        """
        Final shape = (Batch, 1)
        """
        s_emb = self.s_net(s)
        a_emb = self.a_net(a)

        x = torch.cat([s_emb,a_emb], axis=1)
        x = self.act(self.BN1(self.layer1(x)))
        x = self.act(self.BN2(self.layer2(x)))
        q = self.layer3(x)
        return q

    def s_net(self, state):
        """
        Final shape = (Batch, 300)
        """
        B = state.shape[0]
        state = state.view(-1, self.dim)
        s_emb = self.act(self.s_layer1(state))
        s_emb = self.act(self.s_layer2(s_emb))
        s_emb = s_emb.view(B, -1)
        return s_emb
    
    def a_net(self, action):
        """
        Final shape = (Batch, 64)
        """
        action = action.swapaxes(1, 2)
        a_emb = self.act(self.a_layer1(action))
        a_emb = self.act(self.a_layer2(a_emb))
        a_emb = a_emb.squeeze(-1)
        return a_emb