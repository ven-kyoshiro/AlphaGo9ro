# -*- coding:utf-8 -*-
import pygame
from pygame.locals import *
import sys
import time
# from simulator import Sim

def draw(pygame, screen, sysfont, pixels, state):
    bp = pixels[0]
    g = pixels[1]
    mar = pixels[2]
    pixels = [bp,g,mar]
    pas = sysfont.render("pass", True, (0,0,0))
    res = sysfont.render("reset", True, (0,0,0))
    br = sysfont.render("brack", True, (0,0,0))
    br_num = sysfont.render(str(state[82]), True, (0,0,0))
    wh = sysfont.render("white", True, (255,255,255))
    wh_num = sysfont.render(str(state[83]), True, (255,255,255))

    pygame.draw.rect(screen, (222,184,135), Rect(bp[0],bp[1],g*8+mar*2,g*8+mar*2))
    screen.blit(br, (15,15))
    screen.blit(br_num, (15,15+50))
    screen.blit(wh, (600-100,15))
    screen.blit(wh_num, (600-100,15+50))

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
            c = 255*(s-1)
            pygame.draw.circle(screen, (c,c,c), (bp[0]+mar+int((i/9))*g, bp[1]+mar+(i%9)*g), 17)


def convert_to_num(pixels,x,y):
    bp = pixels[0]
    g = pixels[1]
    mar = pixels[2]
    print(x)
    print(y)
    if bp[0] < x and x < bp[0]+g*8+mar*2 and bp[1] < y and y < bp[1]+g*8+mar*2:  # 盤の上か
        print('bannoue')
        #クリック座標を数式で番号に変換
    elif 600-10-80 < x and x < 600-10 and 400-40-15 < y and y < 400-15:  # reset
        print("reset")
    elif 10 <x and x< 10+80 and int(400/2-40/2)+100 < y and y < int(400/2-40/2)+100+40:  # pass1
        print("pass1")
    elif 600-10-80 < x and x < 600 - 10 and int(400/2-40/2)+100 <y and y < int(400/2-40/2)+100 +40:  # pass2
        print("pass2")
    else:
        print('just clicked')


def main():
    # preparation of pygame　
    pygame.init() # 初期化
    bp = [115,15]
    g = 40
    mar = 20
    pixels = [bp,g,mar]
    screen = pygame.display.set_mode((600, 400)) # ウィンドウサイズの指定
    pygame.display.set_caption("GoSimulator") # ウィンドウの上の方に出てくるアレの指定
    sysfont = pygame.font.SysFont(None, 40)

    '''
    # preparation of simulator    
    s = SIM()
    s.set_s()  # 引数のなしの時は何も置かれていない状態となる
    '''

    while True:
        screen.fill((0,100,0,)) # 背景色の指定。
        # draw(pygame, s.get_s)
        draw(pygame, screen, sysfont, pixels, [i%2+1 for i in range(84)])
        pygame.display.update() # 画面更新
        for event in pygame.event.get(): # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEMOTION:
                x, y = event.pos
            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                convert_to_num(pixels,x,y)
if __name__ == '__main__':
    main()
