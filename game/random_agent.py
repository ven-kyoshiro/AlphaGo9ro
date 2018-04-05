# -*- coding:utf-8 -*-
from simulator_test import Sim
import numpy as np
import random 
IS_DEBUG = False
class RandomAgent:
    def __init__(self):
        pass

    def act(self,reshape_self,reshape_opp, reshape_ban, kou):
        # 自分の中で環境を再現してみる
        simsim = Sim()
        state = np.array([0. for i in range(84)])
        ban = 2 - int(reshape_ban)
        state = ban * reshape_self + (3.-ban) * reshape_opp
        simsim.set_s(state,ban,kou)        
        regal = simsim.regal_acts()
        if IS_DEBUG: print(regal)
        a = random.choice(regal)
        if IS_DEBUG: print(a)
        return a
