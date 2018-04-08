# -*- coding:utf-8 -*-
import sys
import time
from tqdm import tqdm
from simulator_test import Sim
from random_agent import RandomAgent
from reinforce_agent import ReinforceAgent
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
    NUM_OF_LEARN = 100
    ai = RandomAgent()
    s = Sim()
    all_records = []

    # NN関係
    nn_kun = ReinforceAgent(restore_call = False)

    # NNを動かします
    with tf.Session() as sess: # if TASK == 'train':              # add in option-2 case
        sess.run(nn_kun.init)                     # option-1
        if nn_kun.restore_call:
            # Restore variables from disk.
            nn_kun.saver.restore(sess, chkpt_file)
        # 100回学習
        for i in range(NUM_OF_LEARN):
            # 100試合対戦
            for j in tqdm(range(NUM_OF_GAMES)):
                records = []
            # preparation of simulator    
                s.reset_s()  # 引数のなしの時は何も置かれていない状態となる
                while True:
                    # ban を上書きする前に，勝敗を保存する？
                    reshape_self, reshape_opp, reshape_ban, kou = s.get_s()
                    
                    if reshape_ban != 2:
                        act_num = nn_kun.act(reshape_self, reshape_opp, reshape_ban, kou )
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


            # データ整理
            batch_xs =
            batch_ys =

            # 学習
            print('\n Training...')
            # 100個選んで学習させる
            nn_kun.train_step.run({nn_kun.x: batch_xs, nn_kun.y_: batch_ys,
                nn_kun.phase_train: True})
            # 途中経過を見る
            if j % 100 == 0:
                cv_fd = {nn_kun.x: batch_xs, nn_kun.y_: batch_ys,
                                               nn_kun.phase_train: False}
                train_loss = nn_kun.loss.eval(cv_fd)
                print('---test---')
                print('[dbg] batch_ys[0]*5 = {0}'.format(np.round([b*5 for b in batch_ys[0]],2)))
                print('[dbg] y_pred[0]*5   = {0}'.format(np.round([b*5 for b in nn_kun.y_pred.eval(cv_fd)[0]],2)))
                train_accuracy = nn_kun.accuracy.eval(cv_fd)

                print('  step, loss, accurary = %6d: %8.4f, %8.4f' % (i,
                    train_loss, train_accuracy)) 

        # 学習後にrandom と 先手50,後手50試合やって結果を表示
        # reinforce 先手
        for j in tqdm(range(50)):
            records = []
            # preparation of simulator    
            s.reset_s()  # 引数のなしの時は何も置かれていない状態となる

            while True:
                # ban を上書きする前に，勝敗を保存する？
                reshape_self, reshape_opp, reshape_ban, kou = s.get_s()
                
                if reshape_ban != 2:
                    if reshape_ban == 1:
                        act_num = nn_kun.act(reshape_self, reshape_opp, reshape_ban, kou )
                    else:
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
            rec = [records[0][-1] for records in all_records]
            kuro_win = NUM_OF_GAMES / 2 + sum(rec) / 2
            siro_win = NUM_OF_GAMES - kuro_win
            print('[reinfoce]黒の勝ち:{0}回, [random]   白の勝ち:{1}回'.format(kuro_win, siro_win))
        # random 先手
        for j in tqdm(50):
            records = []
            # preparation of simulator    
            s.reset_s()  # 引数のなしの時は何も置かれていない状態となる

            while True:
                # ban を上書きする前に，勝敗を保存する？
                reshape_self, reshape_opp, reshape_ban, kou = s.get_s()
                
                if reshape_ban != 2:
                     if reshape_ban == 1:
                        act_num = ai.act(reshape_self, reshape_opp, reshape_ban, kou )
                    else:
                        act_num = nn_kun.act(reshape_self, reshape_opp, reshape_ban, kou )

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
            rec = [records[0][-1] for records in all_records]
            kuro_win = NUM_OF_GAMES / 2 + sum(rec) / 2
            siro_win = NUM_OF_GAMES - kuro_win
            print('[random]  黒の勝ち:{0}回, [reinforce]白の勝ち:{1}回'.format(kuro_win, siro_win))


    # 最後の最後はモデルを保存
    save_path = nn_kun.saver.save(sess, chkpt_file)


if __name__ == '__main__':
    main()
