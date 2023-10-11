import torch
import random
import torch.nn as nn 
import numpy as np
import pandas as pd

from collections import deque
from backtester import BackTester
from torch.optim import Adam
from torch.optim import SGD
from torch.nn import MSELoss
from network import Mask
from network import Rnet
from network import Qnet
from network import Policy

torch.set_printoptions(sci_mode=False)

device = 'cpu'

class RLSEARCH(BackTester):
    def __init__(self, config):
        BackTester.__init__(self, config)

        dim = config['Dim']
        self.mnet = Mask(dim).to(device)
        self.rnet = Rnet(dim).to(device)
        self.qnet = Qnet(dim).to(device)
        self.policy = Policy(dim).to(device)
        self.mse = MSELoss()

        self.opt_r = Adam(self.rnet.parameters(), lr=1e-4)
        self.opt_a = Adam(self.mnet.parameters(), lr=2e-3)
        self.opt_q = Adam(self.qnet.parameters(), lr=1e-4)
        self.opt_p = Adam(self.policy.parameters(), lr=5e-3)
        self.states = None
    
    def save(self):
        torch.save(self.mnet.state_dict(), 'mnet.pth')
        torch.save(self.rnet.state_dict(), 'rnet.pth')
        torch.save(self.qnet.state_dict(), 'qnet.pth')
        torch.save(self.policy.state_dict(), 'policy.pth')

    def get_s(self):
        """
        State로 사용할 데이터 올려놓기
        """
        if self.states is None:
            self.states = [self.get_ScoreEACH(date) \
                        for date in self.universe.index]
            self.states = torch.tensor(np.array(self.states))

    def get_w(self, noise=True):
        """
        Policy로부터 팩터 가중치 샘플링
        """
        return self.mnet.sample(noise).cpu()

    def get_r(self, result:dict):
        """
        결과 메트릭으로부터 reward 계산
        """
        reward = result['sharpe']
        reward = torch.tensor([reward])
        return reward

    def get_R(self, rewards:list):
        """
        순차적으로 받은 reward로 return 계산
        """
        returns = []
        state_return = 0

        for reward in reversed(rewards):
            state_return = reward + 0.95*state_return
            returns.append(state_return)

        returns = returns[::-1]
        returns = torch.tensor(returns)
        returns = torch.unsqueeze(returns, -1)
        return returns
    
    def binning(self, df):
        mapping = self.policy(torch.tensor(df.to_numpy()).unsqueeze(0))
        mapping = torch.squeeze(mapping, 0).detach()
        mapping = pd.DataFrame(mapping.numpy(), index=df.index)
        return mapping
        
    def update(self, w, r, noise, rewards):
        """
        DDPG 스타일 업데이트
        """
        n = noise.detach()
        s = self.states
        action = self.policy(s)
        action = (action * n).float()
        action = torch.sum(action, axis=-1, keepdim=True)

        # Q network update
        R = self.get_R(rewards).float()
        q_hat = self.qnet(s.float(), action.detach())
        q_loss = self.mse(q_hat, R)
        
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # Policy update
        p_loss = -(self.qnet(s.float(), action)).mean()
        self.opt_p.zero_grad()
        p_loss.backward()
        self.opt_p.step()

        # R network update
        r_hat = self.rnet(w.detach())
        r_loss = self.mse(r_hat, r)
        
        self.opt_r.zero_grad()
        r_loss.backward()
        self.opt_r.step()

        # Policy update
        w_loss = -(self.rnet(w)).mean()

        self.opt_a.zero_grad()
        w_loss.backward(retain_graph=True)
        self.opt_a.step()
        return r_loss.item(), w_loss.item()
        
    def search(self, iter, start='1990', end='2024'):
        """
        RL 에이전트 학습 Loop
        """
        
        w_tensor = deque(maxlen=100)
        r_tensor = deque(maxlen=100)
        score = 0
        batch_size = 32

        for i in range(iter):
            weight = self.get_w()

            self.init(weight.detach().numpy(), 
                      start, end, self.binning)
            
            rewards, result = self.test()[-2:]
            reward = self.get_r(result)
            self.get_s()        

            score += 0.01 * (reward.item() - score)
            w_tensor.append(weight)
            r_tensor.append(reward)

            if len(w_tensor) >= batch_size:
                w_batch = random.sample(w_tensor, batch_size)
                r_batch = random.sample(r_tensor, batch_size)

                w_batch = torch.stack(w_batch).float().to(device)
                r_batch = torch.stack(r_batch).float().to(device)
                
                r_loss, w_loss = self.update(w_batch, r_batch, weight, rewards)

                print(f'iter:{i}')
                print(f'reward:{reward.item()}')
                print(f'score:{score}')
                print(f'reward:{reward}')
                print(f'sigma:{self.mnet.sigma}')
                print(f'r loss:{r_loss}')
                print(f'w loss:{w_loss}')
                print(f'points:{self.policy.state_dict()}')
                print(f'{weight.detach()}')
                print(f'{self.get_w(False).detach()}\n')

                

class RANDOMSEARCH(BackTester):
    def __init__(self, config):
        BackTester.__init__(self, config)

        self.dim = config['Dim']
        self.optimal = None

    def get_w(self):
        """
        랜덤 가중치를 리턴
        """
        w = np.random.rand(self.dim)
        w[np.argsort(w)[:8]] = 0.0 
        w = w / np.sum(w)
        return w
    
    def search(self, iter, start='1990', end='2024'):
        """
        랜덤 써치를 통한 최적 가중치 탐색
        """

        best = 0

        for i in range(iter):
            weight = self.get_w()
            self.init(weight, start, end)
            result = self.test()[-1] 
            reward = result['sharpe'] 

            self.optimal = weight \
                if reward > best else self.optimal
            
            best = reward \
                if reward > best else best
            
            print(f'iter:{i}')
            print(f'best:{best}\n')
        
    def save(self, path):
        param = torch.tensor(self.optimal)
        torch.save(param, path)