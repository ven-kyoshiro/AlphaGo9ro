# -*- coding:utf-8 -*-
import sys
import time
from tqdm import tqdm
from simulator_test import Sim
from random_agent import RandomAgent
import csv

def is_black_win(sim):
    bl_v, wh_v = sim.get_eval()
    b_score = float(len(bl_v))
    w_score = float(len(wh_v))+ 6.5 # 六目半
    for i in sim.state[1:82]:
        if i == 1.:
            b_score += 1
        elif i == 2.:
            w_score += 1
    return b_score > w_score

def w_fnc(data,name):
    with open(name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)

def main():
    NUM_OF_GAMES = 100
    ai = RandomAgent()
    s = Sim()
    all_records = []
    for i in tqdm(range(NUM_OF_GAMES)):
        records = []
    # preparation of simulator    
        s.reset_s()  # 引数のなしの時は何も置かれていない状態となる
        while True:
            # ban を上書きする前に，勝敗を保存する？
            reshape_self, reshape_opp, reshape_ban, kou = s.get_s()
            
            if reshape_ban != 2:
                act_num = ai.act(reshape_self, reshape_opp, reshape_ban, kou )
                s.act(act_num)
            else:
                # 黒の勝ち+1,負け-1
                outcome = 2*is_black_win(s)-1
                break
            records.append([reshape_self[1:], reshape_opp[1:], reshape_ban, act_num])    
        #outcomeを付け足し
        for j in range(len(records)):
            records[j].append((1.-2.*(j%2))*outcome)
        all_records.append(records)
    # stackしたレコードを書き出す
    w_fnc(all_records,'all_records.csv')
    rec = [records[0][-1] for records in all_records]
    kuro_win = NUM_OF_GAMES / 2 + sum(rec) / 2
    siro_win = NUM_OF_GAMES - kuro_win
    print('黒の勝ち:{0}回, 白の勝ち:{1}回'.format(kuro_win, siro_win))

if __name__ == '__main__':
    main()
