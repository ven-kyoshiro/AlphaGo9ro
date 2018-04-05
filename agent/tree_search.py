# -*- coding:utf-8 -*-
from simulator import sim
from networks import model
import math
import numpy as np
import random
'''
とりあえず試しにデータ構造を作ってみる
ゲーム木のデータ構造
key = str(act_num)+str(sを一次元配列で起こしたもの)
(0:なし，1：自分，2:相手)


今いるエッジは外で保存するようにした方が良いかも
'''
game_tree = {}

class game_tree:
    def __init__(self,s):
        self.c_puct = 3. # hyper palam
        self.gt = {}
        # 現在見ているノードに関する情報
        self.search_sim = sim.simulator(s)#TODO
        self.s = s
        self.s_name = self.s_to_name(s)# 自分がどこにいるかを表す
    def eval_ini_state(self):
        # モデルでsを評価して，N(int),W,Q,P,V,をそれぞれで初期化
        _, p = self.eval_state(s)
        for act_num in range(0,82):
            self.gt[str(act_num)+self.s_name]  =  {'parent':'root','N':0,'W':0.0,'Q':0.0,'P':9999.9}
            #TODO: もしかしたら自分のkeyをメモっておく必要が出てくるかも
    def eval_state(self):
        i = random.choice(range(1,9))
        s = self.d_rot_ref(i,s)
        v,p = model.eval_state(s)
        p = self.rot_ref(i,p)
        return v,p
# wip
    def select(self):
        not_root = True
        while not_leaf:
            U_Q_argmax = self.select_sub()
            # if 辞書で引ける→そのまま
                # pass
            # else モデル評価のqueに入れて進んだ先を初期化そ，そのあと遷移させる
                v, p = self.eval_state(s)   
                # 今の状態に対して，可能なノードを評価
                # ここはqueに入れて応答をマツ()
                v,p = self.eval_state()
                self.gt[str(act_num)+self.s_name]  =  {'parent':#TODO'root','N':0,'W':0.0,'Q':0.0,'P':9999.9}
                not_leaf = False
                # 次はbackup    
    def back_up(self):
        # selectで選んだノードを親順に辿って更新していく
        # parent がrootだった場合アルゴリズム終了でselectステップに戻る
    def play(self):
        # gt[act_num + 'root']を全部読み出して，サンプリングする
# 外のルーチンで，サンプリングしたら，rootの位置を変えてもう一度実行　ゲームの終端まで続ける
           
    def select_sub(self): # 現在のノードから
        root_sum =  math.sqrt(sum([self.gt[str(i)+self.s_name]['N'] for i in range(0,82)]))
        U_Q = np.array([self.ucb1(self.gt[str(i)+self.s_name],root_sum) for i in range(0,82)]])
        U_Q_argmax = np.where(U_Q == U_Q.max())[0]
        if len(U_Q_argmax)>1: 
            U_Q_argmax = random.choice(U_Q_argmmax)
        else:
            U_Q_argmax = U_Q_argmax[0]
        return U_Q_argmax

    def s_to_name(self,s):
        #wip 
    def ucb1(self,values,root_sum):
        return values['Q'] + self.c_puct*values['P']*root_sum/(1.+values['N'])
    def rot_ref(self,i,s):
        # wip
    def inv_rot_ref(self,i,s):
        # wip
