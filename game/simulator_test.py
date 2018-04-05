# -*- coding:utf-8 -*-
import numpy as np
import copy

class Sim:
    def __init__(self):
        self.st2ind={} #ket=state_indx,value = list_indx TODO:この関数initへ
        self.place2id = [] # TODO:この関数initへ
        self.was_pass = False
        count = 1
        for i in range(11):
            place2id_sub = []
            for j in range(11):
                if i*(10-i)*j*(10-j) ==0:
                    place2id_sub_sub = -1
                else:
                    self.st2ind[count] = [i,j]
                    place2id_sub_sub = count
                    count+=1
                place2id_sub.append(place2id_sub_sub)
            self.place2id.append(place2id_sub)


    def set_s(self):
        self.state = np.array([0. for i in range(84)])
        self.ban = 1.
        self.kou = []
        self.game_over = False

    def get_s(self):
        if self.game_over:
            self.ban = 0
        ## 扱いやすくるすために盤面のデータを整形
        reshape_ban = 2 - self.ban
        stt = self.state[0:82]
        if reshape_ban == 0: # 白番の時
            reshape_self = stt*(stt-1)/2 # 2だけが1
            reshape_opp = stt*(2-stt) # 1だけが1
        else:
            reshape_opp = stt*(stt-1)/2 # 2だけが1
            reshape_self = stt*(2-stt) # 1だけが1           
        return reshape_self,reshape_opp,reshape_ban

    def is_enclosed(self,act_num):
        # その石が死んでいるか確認
        pos = [self.st2ind[act_num][0],self.st2ind[act_num][1]]
        #TODO dbg
        # print(pos)
        # print(act_num)
        self.ban = 3. - self.ban
        find_table = self.get_find_table()
        find_table[pos[0]][pos[1]]= 3. - self.ban
        is_del = len(self.can_get(find_table,pos)) != 0
        self.ban = 3 - self.ban
        return is_del

    def is_kou(self,act_num):
        if act_num not in self.kou:
            return False
        self.ban = 3.0 - self.ban
        find_table_rev = self.get_find_table()
        self.ban = 3.0 - self.ban
        # num を場所に直して，４方向が全て相手でかつ
        for j in [[0,-1],[0,1],[-1,0],[1,0]]:
            # 隣接マスが相手の駒の時
            pos = [self.st2ind[act_num][0]+j[0],
                    self.st2ind[act_num][1]+j[1]]
            if find_table_rev[pos[0]][pos[1]] != 3. - self.ban:
                return False
        return True

    def regal_acts(self):
        # すでに石
        regal = [0]
        for i in range(1,82):
            if self.state[i] != 0.:  # already placed
                continue
            else: # 置ける
                if self.is_enclosed(i):  # 着手禁止点
                    find_table = self.get_find_table()
                    find_table[self.st2ind[i][0]][self.st2ind[i][1]] = self.ban
                    if len(self.get_all_can_get_ids(find_table,i)) > 0:  # 取れる
                        if self.is_kou(i): # こうか
                            continue
                        else: # こうでない
                            regal.append(i)
                    else: # 取れない
                        continue
                else: # 着手禁止点でない
                    regal.append(i)
        return regal

    def get_find_table(self):
        find_table = []
        count = 1
        for i in range(11):
            ft_sub=[]
            for j in range(11):
                if i*(10-i)*j*(10-j) ==0:
                    ft_sub_sub=self.ban
                else:
                    ft_sub_sub=self.state[count]
                    count+=1
                ft_sub.append(ft_sub_sub)
            find_table.append(ft_sub)
        return find_table

    def can_get(self,find_table,pos):
        can_get_ids = []
        print('[find]->step to next stone x1:'+str(pos[0])+',y:'+str(pos[1]))
        not_del = self.find(find_table,pos)
        if not not_del:
            print('[find]delete stones')
            new_state = np.array([0.]+\
                [find_table[self.st2ind[i][0]][
                            self.st2ind[i][1]] for i in range(1,82)])
            for i in range(len(new_state)):
                if new_state[i] == 3.:
                    can_get_ids.append(i)
        return can_get_ids

    def get_all_can_get_ids(self,find_table,act_num):
        can_get_ids = []
        ft = copy.deepcopy(find_table)
        print('[base]x:'+str(self.st2ind[act_num][0])+' y:'+str(self.st2ind[act_num][1]))
        find_table[self.st2ind[act_num][0]][self.st2ind[act_num][1]] = self.ban
        for j in [[0,-1],[0,1],[-1,0],[1,0]]:
            ft = copy.deepcopy(find_table)
            # 隣接マスが相手の駒の時
            pos = [self.st2ind[act_num][0]+j[0],
                   self.st2ind[act_num][1]+j[1]]
            if ft[pos[0]][pos[1]]== 3. - self.ban:
                can_get_ids += self.can_get(ft,pos)
        return can_get_ids

    def find(self,ft,l_id):
        ft[l_id[0]][l_id[1]] = 3.
        for i in [[0,-1],[0,1],[-1,0],[1,0]]:
            #TODO:debug
            print('[check]x: '+str(l_id[0]+i[0])+',y: '+str(l_id[1]+i[1]))
            
            if ft[l_id[0]+i[0]][l_id[1]+i[1]] == 3.0 - self.ban:
                print('[find]-->step to next stone')
                not_del = self.find(ft,[l_id[0]+i[0],l_id[1]+i[1]])
                if not_del:
                    return True
            elif ft[l_id[0]+i[0]][l_id[1]+i[1]] == 0.:
                print('[find]-->survive')
                return True
            else:
                print('[find]-->stop')

    def act(self,act_num):
        if 0<act_num:
            self.state[act_num] = self.ban
            find_table = self.get_find_table()
            # 上下左右の順に確認
            can_get_ids = self.get_all_can_get_ids(find_table, act_num)
            #TODO 帰ってきた部分を置き換える
            for i in can_get_ids:
                self.state[i] = 0.
            self.kou = can_get_ids
            self.state[81+int(self.ban)]+=len(self.kou)
            print('show find_table below')
            for ft in find_table:
                print(ft)
            self.was_pass = False
        if act_num ==0:
            if self.was_pass == True:
                self.game_over = True
                self.was_pass = False
            else:
                self.was_pass = True
        self.ban = 3. -self.ban
    
    def fill_area(self, find_table, pos):
        find_table[pos[0]][pos[1]] = 3.
        self.my_pos.append(pos)
        for i in [[0,-1],[0,1],[-1,0],[1,0]]:
            pos_i = find_table[pos[0]+i[0]][pos[1]+i[1]]
            if  pos_i == 3. - self.ban:
                self.my_flug = False
            elif pos_i == 0.:
                self.fill_area(find_table,[pos[0]+i[0],pos[1]+i[1]])

    def get_bans_pos(self,ban):
        self.ban = ban
        find_table = self.get_find_table()
        bans_pos = []
        for i in range(1,82):
            pos = self.st2ind[i]
            if find_table[pos[0]][pos[1]] == 0.:
                self.my_flug = True
                self.my_pos = []
                self.fill_area(find_table,pos)
                if self.my_flug:
                    bans_pos += self.my_pos
                    for dp in self.my_pos:
                        find_table[dp[0]][dp[1]] = -1. * self.ban
                else:
                    for dp in self.my_pos:
                        find_table[dp[0]][dp[1]] = -5.
        return bans_pos

    def get_eval(self):
        ban_memory = self.ban
        bl = self.get_bans_pos(1.0)
        bl = [self.place2id[b[0]][b[1]] for b in bl]
        wh = self.get_bans_pos(2.0)
        wh = [self.place2id[w[0]][w[1]] for w in wh]
        self.ban = ban_memory
        return bl,wh

