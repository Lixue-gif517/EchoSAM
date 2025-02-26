import pandas as pd
import cv2
import numpy as np

import math


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

import csv
with open('KeyPoints.csv', mode='w', newline='') as file:  # 单独运行
    writer = csv.writer(file)
    writer.writerow(["imagename", "class", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y"])


def get_one_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    mask_170 = np.where(mask == 170, 1, 0)
    mask_85 = np.where(mask == 85, 1, 0)
    mask_255 = np.where(mask == 255, 1, 0)
    mask_1 = np.where(mask_170 == 1, 255, mask_170).astype( np.uint8 )
    mask_2 = np.where(mask_85 == 1, 255, mask_85).astype( np.uint8 )
    mask_3 = np.where(mask_255 == 1, 255, mask_255).astype( np.uint8 )
    # 保存二值图像
    # cv2.imwrite("lv_epi.png", mask_1)
    # cv2.imwrite("lv.png", mask_2)
    # cv2.imwrite("la.png", mask_3)
    return mask_1, mask_2, mask_3

def get_edges_points(contours):
    edges_points_l = []
    for i in contours:
        for point in i:
            x,y = point[0]
            edges_points_l.append((x,y))
            # print((x,y))
    return edges_points_l

# image = cv2.imread('./datasets/CAMUS/label/patient0001_2CH_ED.png',0)
def get_keypoints(imagepath):
    df = pd.read_csv('KeyPoints.csv')
    basename =  imagepath.split('/')[-1]
    mask_1, mask_2, mask_3 = get_one_mask(imagepath)
    edges_1 = cv2.Canny(mask_1, 100, 200)
    contours_1,_ = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # myo
    # cv2.imshow('Edges', edges_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edges_2 = cv2.Canny(mask_2, 100, 200)
    contours_2,_ = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # lv

    edges_3 = cv2.Canny(mask_3, 100, 200)
    contours_3,_ = cv2.findContours(edges_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # la

    # cv2.drawContours(mask_1, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Image with Contours", mask_1)
    edges_points_lv = get_edges_points(contours_2)
    edges_points_la = get_edges_points(contours_3)
    edges_points_myo = get_edges_points(contours_1)
    same = []
    mindis = 1000000  # 保存最小
    same1 = []
    mindis1 = 1000000  # 保存第二小
    lv_x_l = []
    lv_y_l = []
    la_x_l = []
    la_y_l = []
    for (lv_x,lv_y) in edges_points_lv:   # 时间复杂度太高！！！
        lv_x = int(lv_x)
        lv_y = int(lv_y)
        lv_x_l.append(lv_x)
        lv_y_l.append(lv_y)
        for (la_x,la_y) in edges_points_la:
            la_x = int(la_x)
            la_y = int(la_y)
            la_x_l.append(la_x)
            la_y_l.append(la_y)
            for (myo_x,myo_y) in edges_points_myo:
                # mindis1 = calculate_distance(lv_x,lv_y,la_x,la_y)
                # mindis2 = calculate_distance(lv_x, lv_y, myo_x, myo_y)
                myo_x = int(myo_x)
                myo_y = int(myo_y)
                mindis3 = abs(myo_x-lv_x) + abs(la_y-lv_y) + abs(myo_x-la_x) + abs(myo_y-lv_y)
                if mindis3 <= mindis:  # mindis3是最小的点
                    mindis1 = mindis
                    same1 = same
                    mindis = mindis3
                    same = [lv_x,lv_y,la_x,la_y,myo_x,myo_y]
                    # same.append((lv_x,lv_y,la_x,la_y,myo_x,myo_y))
    p3_y = min(lv_y_l)
    index_ = lv_y_l.index(p3_y)
    p3_x = lv_x_l[index_]

    print(same)

    p3_y_la = max(la_y_l)
    index_la = la_y_l.index(p3_y_la)
    p3_x_la = la_x_l[index_la]

    if same[0] < same1[0]:
        lv_row = {"imagename":basename,"class":"lv","p1_x":same[0],"p1_y":same[1],"p2_x":same1[0],
                  "p2_y":same1[1],"p3_x":p3_x,"p3_y":p3_y}
        myo_row = {"imagename": basename, "class": "myo", "p1_x": same[4], "p1_y": same[5], "p2_x": same1[4],
                  "p2_y": same1[5], "p3_x": p3_x, "p3_y": p3_y}
        la_row = {"imagename": basename, "class": "la", "p1_x": same[2], "p1_y": same[3], "p2_x": same1[2],
                  "p2_y": same1[3], "p3_x": p3_x_la, "p3_y": p3_y_la}
    else:
        lv_row = {"imagename":basename,"class":"lv","p1_x":same1[0],"p1_y":same1[1],"p2_x":same[0],
                  "p2_y":same[1],"p3_x":p3_x,"p3_y":p3_y}
        myo_row = {"imagename": basename, "class": "myo", "p1_x": same1[4], "p1_y": same1[5], "p2_x": same[4],
                  "p2_y": same[5], "p3_x": p3_x, "p3_y": p3_y}
        la_row = {"imagename": basename, "class": "la", "p1_x": same1[2], "p1_y": same1[3], "p2_x": same[2],
                  "p2_y": same[3], "p3_x": p3_x_la, "p3_y": p3_y_la}

    # common_elements = list(edges_points_lv & edges_points_la)
    df.loc[len(df)] = lv_row
    df.loc[len(df)] = myo_row
    df.loc[len(df)] = la_row
    df.to_csv('KeyPoints.csv', index=False)
    # print(len(edges_points_lv), len(edges_points_la), len(edges_points_myo))


df = pd.read_csv('KeyPoints.csv')
import os
directory_name = "./datasets/CAMUS/label"
for filename in os.listdir(directory_name):
    print(filename)
    get_keypoints(directory_name + "/" + filename)
df.to_csv('KeyPoints.csv', index=False)