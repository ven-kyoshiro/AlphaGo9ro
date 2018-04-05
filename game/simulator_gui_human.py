# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys
import time
from simulator_test import Sim

def draw(pygame, screen, sysfont, pixels, state, ban, x, y, bl_v, wh_v):
    bp = pixels[0]
    g = pixels[1]
    mar = pixels[2]
    pixels = [bp,g,mar]
    b_score = len(bl_v) 
    w_score = len(wh_v)+ 6.5 # 六目半
    # 勝敗のカウント
    for i in state[1:82]:
        if i == 1.:
            b_score += 1
        elif i == 2.:
            w_score += 1
    pas = sysfont.render("pass", True, (0,0,0))
    res = sysfont.render("reset", True, (0,0,0))
    bl = sysfont.render("black", True, (0,0,0))
#    bl_num = sysfont.render(str(state[82]), True, (0,0,0))
    bl_num = sysfont.render(str(b_score), True, (0,0,0))
    wh = sysfont.render("white", True, (255,255,255))
#    wh_num = sysfont.render(str(state[83]), True, (255,255,255))
    wh_num = sysfont.render(str(w_score), True, (255,255,255))

    pygame.draw.rect(screen, (222,184,135), Rect(bp[0],bp[1],g*8+mar*2,g*8+mar*2))
    screen.blit(bl, (15,15))
    screen.blit(bl_num, (15,15+50))
    screen.blit(wh, (600-100,15))
    screen.blit(wh_num, (600-100,15+50))

    if ban == 1.:
        pygame.draw.rect(screen, (0,0,0), Rect(15-4,15-3,82,34),3)
    elif ban == 2.:
        pygame.draw.rect(screen, (255,255,255), Rect(600-100-4,15-3,82,34),3)
    else:
        if b_score>w_score:
            pygame.draw.rect(screen, (0,0,0), Rect(15-4,15-3,82,34),10)
        else:
            pygame.draw.rect(screen, (255,255,255), Rect(600-100-4,15-3,82,34),10)

    pygame.draw.rect(screen, (255,69,0), Rect(600-10-80,400-40-15,80,40))
    screen.blit(res, (8+600-10-80,5+400-40-15))
    pygame.draw.rect(screen, (127,127,127), Rect(10,int(400/2-40/2)+100,80,40))
    screen.blit(pas, (10+10,5+int(400/2-40/2)+100))
    pygame.draw.rect(screen, (127,127,127), Rect(600-10-80,int(400/2-40/2)+100,80,40))
    screen.blit(pas,(600-10-80+10,5+int(400/2-40/2)+100))

    for i in range(8):
        for j in range(8):
            pygame.draw.rect(screen, (0,0,0), Rect(bp[0]+mar+i*g,bp[1]+mar+j*g ,g,g),1)

    for i, s in enumerate(state[1:82]):
        if s != 0:
            c = 255*(s - 1)
            pygame.draw.circle(screen, (c,c,c), (bp[0]+mar+(i%9)*g, bp[1]+mar+int(i/9)*g), 17)
        if i+1 in bl_v:
            c  = 20
            pygame.draw.circle(screen, (c,c,c), (bp[0]+mar+(i%9)*g, bp[1]+mar+int(i/9)*g), 5)
        if i+1 in wh_v:
            c = 235
            pygame.draw.circle(screen, (c,c,c), (bp[0]+mar+(i%9)*g, bp[1]+mar+int(i/9)*g), 5)
        if i+1 in bl_v and i+1 in wh_v:
            c = 128
            pygame.draw.circle(screen, (c,c,c), (bp[0]+mar+(i%9)*g, bp[1]+mar+int(i/9)*g), 5)
    pygame.draw.circle(screen, (255,0,0), (x, y),  5)

def convert_to_num(pixels,x,y):
    bp = pixels[0]
    g = pixels[1]
    mar = pixels[2]
    print('[gui]('+str(x)+','+str(y)+') was clicked')
    if bp[0] < x and x < bp[0]+g*8+mar*2 and bp[1] < y and y < bp[1]+g*8+mar*2:  # 盤の上か
        print('[gui]-> bannoue')
        num = (int((x-bp[0])/g)+1) + 9 * int((y-bp[1])/g)
        print('[gui]-->clicked index is '+str(num))
        #クリック座標を数式で番号に変換
    elif 600-10-80 < x and x < 600-10 and 400-40-15 < y and y < 400-15:  # reset
        num = -1
        print("[gui]->reset")
    elif 10 <x and x< 10+80 and int(400/2-40/2)+100 < y and y < int(400/2-40/2)+100+40:  # pass1
        num = 0
        print("[gui]->pass1")
    elif 600-10-80 < x and x < 600 - 10 and int(400/2-40/2)+100 <y and y < int(400/2-40/2)+100 +40:  # pass2
        num = 0
        print("[gui]->pass2")
    else:
        num = -2
        print('[gui]->just clicked')
    return num

def main():
    # preparation of pygame　
    pygame.init() # 初期化
    bp = [115,15]
    g = 40
    mar = 20
    x, y = 0, 0
    pixels = [bp,g,mar]
    screen = pygame.display.set_mode((600, 400)) # ウィンドウサイズの指定
    pygame.display.set_caption("GoSimulator") # ウィンドウの上の方に出てくるアレの指定
    sysfont = pygame.font.SysFont(None, 40)

    # preparation of simulator    
    s = Sim()
    s.reset_s()  # 引数のなしの時は何も置かれていない状態となる

    while True:
        screen.fill((0,100,0,)) # 背景色の指定。
        # state,ban = s.get_s()
        # gui用に整形をかませる
        reshape_self, reshape_opp, reshape_ban, _ = s.get_s()
        ban = 2 - int(reshape_ban)
        # game over すなわち ban == 0の時の処理
        if ban != 0:
            state = ban * reshape_self + (3.-ban)*reshape_opp
        bl, wh = s.get_eval()
        # TODO for debug
        draw(pygame, screen, sysfont, pixels,state, ban ,x,y,bl,wh)
        pygame.display.update() # 画面更新
        for event in pygame.event.get(): # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN and event.button == 1: 
                x, y = event.pos
                num = convert_to_num(pixels,x,y)
                if num == -1:
                    s.reset_s()
                else:
                    if num in s.regal_acts() and ban != 0:
                        s.act(num)
                        bl, wh = s.get_eval()
                        if len(bl) != len(set(bl)):
                            print('[error] get_eval has over lap')
                
if __name__ == '__main__':
    main()
