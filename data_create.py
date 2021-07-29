import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

#dir_back = 바탕사진 디렉토리 / dir_input = 들어갈 사진 디렉토리
#background_list = 바탕 사진들의 리스트값 / iput_list = 들어갈 사진들의 리스트값 / dir_save = 저장할 디렉토리
#dim = 종양사진 resize 할 크기 설정 / make_image_cnt = 만들고 싶은 갯수 설정

def create_data_list(dir_back,dir_input,background_list, input_list,dir_save,make_image_cnt,dim):

    for i in range(make_image_cnt):

        # normal, stone의 길이만큼 랜덤변수로 생성
        B_cnt = random.randint(0,len(background_list)-1)




        #불러오기
        B_image=cv2.imread(dir_back+'/'+background_list[B_cnt])

        I_cnt = random.randint(0, len(input_list) - 1)
        I_image=cv2.imread(dir_input+'/'+input_list[I_cnt])


        # N_image = cv2.cvtColor(N_image,cv2.COLOR_BGR2RGB)
        #
        # S_image = cv2.cvtColor(S_image,cv2.COLOR_BGR2RGB)


        #종양이 중앙에 배치되게 사이즈 조정

        h, w = B_image.shape[:2]
        B_image_small = B_image[int(w/5):int(4*w/5), int(h/5):int(4*h/5)]
        dh, dw = B_image_small.shape[:2]


        #종양 스크린샷 사이즈 조정
        I_image = cv2.resize(I_image,(dim,dim),interpolation=cv2.INTER_AREA)
        ch, cw = I_image.shape[:2]

        #normal사진안에 랜덤으로 자리잡게 설정
        dx = random.randint(0,dh-ch)
        dy = random.randint(0,dw-cw)

        #랜덤 위치에 종양합성
        B_image_small[dx:dx+ch , dy:dy+cw] =I_image

        B_image[int(w/5):int(4*w/5), int(h/5):int(4*h/5)] = B_image_small

        #이미지 저장
        # for j in range(4):
        #     cv2.imwrite(dir_save+str(i)+"-"+str(j)+".jpg",B_image)
        cv2.imwrite(dir_save + str(i+1)+ ".jpg", B_image)

def create_data_Ones(dir_back,dir_input,background_list, input_list,dir_save,make_image_cnt,dim):

    for i in range(make_image_cnt):

        # normal, stone의 길이만큼 랜덤변수로 생성
        B_cnt = random.randint(0,len(background_list))
        I_cnt = random.randint(0,len(input_list))



        #불러오기
        B_image=cv2.imread(dir_back+'/'+background_list[B_cnt])
        I_image=cv2.imread(dir_input+'/'+input_list[I_cnt])

        print(I_image.shape)
        # N_image = cv2.cvtColor(N_image,cv2.COLOR_BGR2RGB)
        #
        # S_image = cv2.cvtColor(S_image,cv2.COLOR_BGR2RGB)


        #종양이 중앙에 배치되게 사이즈 조정
        B_image_small = B_image[90:360 , 90:360]
        dh, dw = B_image_small.shape[:2]


        #종양 스크린샷 사이즈 조정
        I_image = cv2.resize(I_image,(dim,dim),interpolation=cv2.INTER_AREA)
        ch, cw = I_image.shape[:2]

        #normal사진안에 랜덤으로 자리잡게 설정
        dx = random.randint(0,dh-ch)
        dy = random.randint(0,dw-cw)

        #랜덤 위치에 종양합성
        B_image_small[dx:dx+ch , dy:dy+cw] =I_image

        B_image[90:360, 90:360] = B_image_small

        #이미지 저장
        for j in range(4):
            cv2.imwrite(dir_save+str(i)+".jpg",B_image)



def create_data_normal(dir_back,dir_input,background_list, input_list,dir_save,make_image_cnt):

    for i in range(make_image_cnt):
        # normal, stone의 길이만큼 랜덤변수로 생성
        B_cnt = random.randint(0,len(background_list)-1)
        I_cnt = random.randint(0,len(input_list)-1)


        #불러오기
        B_image=cv2.imread(dir_back+'/'+background_list[B_cnt])
        I_image=cv2.imread(dir_input+'/'+input_list[I_cnt])



        # N_image = cv2.cvtColor(N_image,cv2.COLOR_BGR2RGB)
        #
        # S_image = cv2.cvtColor(S_image,cv2.COLOR_BGR2RGB)


        #종양이 중앙에 배치되게 사이즈 조정
        B_image_small = B_image[90:360 , 90:360]
        dh, dw = B_image_small.shape[:2]

        random_x = random.randint(90,300)
        random_y = random.randint(90,300)

        random_h = random.randint(30,150)
        random_w = random.randint(30,150)

        I_image = I_image[random_x : random_x+random_h, random_y:random_y+random_w]
        ch, cw = I_image.shape[:2]

        #normal사진안에 랜덤으로 자리잡게 설정
        dx = random.randint(0,dh-ch)
        dy = random.randint(0,dw-cw)

        #랜덤 위치에 종양합성
        B_image_small[dx:dx+ch , dy:dy+cw] =I_image

        B_image[90:360, 90:360] = B_image_small

        #이미지 저장
        # for j in range(4):
        #     cv2.imwrite(dir_save+str(i)+"-"+str(j)+".jpg",B_image)
        cv2.imwrite(dir_save + str(i)+ ".jpg", B_image)


# normal, stone데이터 주소설정
dir_normal = "/Users/ShimBoSeok/Desktop/data/normal"
dir_stone = "/Users/ShimBoSeok/Desktop/data/stone"

list_normal = os.listdir(dir_normal)
list_stone = os.listdir(dir_stone)

dir_save = "/Users/ShimBoSeok/Desktop/stone_classification_2020_0703/dataset/SET_SBS/stone_3/result-"

make_image_cnt = 2000
dim = 64

#함수 적용
create_data_list(dir_normal,dir_stone,list_normal,list_stone,dir_save,make_image_cnt,dim)
#create_data_normal(dir_normal,dir_stone,list_normal,list_stone,dir_save,make_image_cnt)

