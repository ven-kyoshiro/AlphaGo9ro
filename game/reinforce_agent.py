# -*- coding:utf-8 -*-
from simulator_test import Sim
import numpy as np
import random
from scipy import stats
import copy
from reinforce_nn import Nn
IS_DEBUG = False
class ReinforceAgent(Nn):
    def __init__(self,restore_call):
        super().__init__(restore_call = False)
        self.memory = np.array([[0. for i in range(81)] for i in 4])
    def act(self,reshape_self,reshape_opp, reshape_ban, kou):
        # メモリを更新
        self.memory[2] = copy.deepcopy(self.memory[0])
        self.memory[3] = copy.deepcopy(self.memory[1])
        self.memory[0] = copy.deepcopy(reshape_self[1:82])
        self.memory[1] = copy.deepcopy(reshape_opp[1:82])
        # 自分の中で環境を再現してみる
        bans = np.array([[reshape_ban]*81])
        simsim = Sim()
        state = np.array([0. for i in range(84)])
        ban = 2 - int(reshape_ban)
        state = ban * reshape_self + (3.-ban) * reshape_opp
        simsim.set_s(state,ban,kou)        
        regal = simsim.regal_acts()
        batch_xs = np.append(self.memory, bans)
        batch_ys = np.array([[0.for i in range(82)]])
        # NNで82次元の確率値を所得　
        #TODO この辺はプリントデバッグ必要かもね〜
        cv_fd = {nn_kun.x: batch_xs, nn_kun.y_: batch_ys,
                                        nn_kun.phase_train: False}
        act_prob = nn_kun.y_pred.eval(cv_fd)[0] #TODO:かっこが一個余分かも
        regal_act_prob = act_prob*(np.identity(82)[regal]).sum(axis = 0)
        if regal_act_prob.sum() == 0.:
            a = random.choice(regal)
            print('[aleat] regal prob is zero')
        else:
            regal_act_prob = regal_act_prob/regal_act_prob.sum()
            xk = np.arange(82)
            pk = regal_act_prob
            if regal_act_prob.sum() !=1.0:
                 actions[regal_act_prob[0]] += 1.0 - regal_act_prob.sum()
            custm = stats.rv_discrete(name='custm', values=(xk, pk))
            a =  custm.rvs(size=1)[0]

        if IS_DEBUG: print(regal)
        if IS_DEBUG: print(a)
        return a
