import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import math

def find_parallel_line_intersection(mask, point, slope,line_length=100):

    x1, y1 = point
    # 计算线段的起始和结束点
    x2 = int(x1 + line_length)
    y2 = int(y1 + slope * (x2 - x1))

    x3 = int(x1 - line_length)
    y3 = int(y1 - slope * (x2 - x1))

    # 在掩码上绘制直线，并查找交点
    line_image = np.zeros_like(mask)
    cv2.line(line_image, (x3, y3), (x2, y2), 255, 1)  # 在黑色背景上绘制白色直线
    cv2.imwrite('line_image.png', line_image)
    # 执行形态学操作以寻找交点
    edges = cv2.Canny(mask, 100, 200)
    intersection = cv2.bitwise_and(edges, line_image)  # 找到掩码和直线的交点

    # 查找交点坐标
    points = np.column_stack(np.nonzero(intersection))

    return points

def calculate_length_from_point(mask, start_point, slope, max_length=1000):
    """
    从指定点出发，沿着给定的斜率方向计算长度，直到遇到掩码的边界

    :param mask: 二值掩码图像 (numpy array)
    :param start_point: 起始点 (x, y)
    :param slope: 直线的斜率
    :param max_length: 最大检查长度
    :return: 计算出的长度
    """
    x, y = start_point
    dx = 1  # 步长沿x轴
    dy = slope  # 步长沿y轴

    length = 0

    for i in range(max_length):
        # 计算当前位置
        new_x = int(x + dx * i)
        new_y = int(y + dy * i)

        # 检查是否超出图像边界
        if new_x < 0 or new_x >= mask.shape[1] or new_y < 0 or new_y >= mask.shape[0]:
            break

        # 检查当前位置是否在掩码中
        if mask[new_y, new_x] > 0:  # 如果是前景像素
            length += np.sqrt(dx**2 + dy**2)  # 计算长度
        else:
            break

    return length



def divide_line_segment(start, end, num_segments=20):
    """
    将一条线段划分为指定数量的段，并返回每个节点的坐标。

    :param start: 线段起点坐标 (x1, y1)
    :param end: 线段终点坐标 (x2, y2)
    :param num_segments: 划分段数
    :return: 每个节点的坐标列表
    """
    # 计算每段的增量
    x_values = np.linspace(start[0], end[0], num_segments + 1)
    y_values = np.linspace(start[1], end[1], num_segments + 1)
    # 生成节点坐标
    nodes = list(zip(x_values, y_values))
    return nodes

def calculate_ejection_fraction(end_diastolic_volume, end_systolic_volume):
    """
    计算射血分数

    :param end_diastolic_volume: 舒张末期体积
    :param end_systolic_volume: 收缩末期体积
    :return: 射血分数
    """
    return (end_diastolic_volume - end_systolic_volume) / end_diastolic_volume

def calculate_V(mask,aim_data):
    x41,y41,x42,y42,x43,y43 = aim_data[0],aim_data[1],aim_data[2],aim_data[3],aim_data[4],aim_data[5]
    p1_4ch = (x41,y41)
    p2_4ch = (x42,y42)
    p3_4ch = (x43,y43)
    p4_4ch = ((x42+x43)/2,(y42+y43)/2)
    L_4ch = math.sqrt(math.pow(((x42+x43)/2-x41),2) + math.pow(((y42+y43)/2 - y41), 2))
    d = L_4ch/21
    # 计算划分20段节点坐标
    nodes_4ch = divide_line_segment(p1_4ch, p4_4ch, 20)

    slope_4ch = (y43-y42)/(x43-x42)
    V_ed = 0
    len_list = []  # ai bi
    for node_4ch in nodes_4ch:
        length = calculate_length_from_point(mask, node_4ch, slope_4ch)
        len_list.append(length)
        # intersections_4ch = find_parallel_line_intersection(mask_end_diastole, node_4ch, slope_4ch)
    for a in len_list:
        V_seg = d * math.pi * a * a
        V_ed = V_ed + V_seg
    # print(V_ed/21)
    return V_ed/21  # 返回容积


if __name__ == '__main__':
    # 创建一个示例二值掩码
    mask_end_diastole  = cv2.imread("C:/Users/Sheila/Desktop/DLUT/Medical_Seg/DataSets/EchoNet-Dynamic/mask/0X1A2A76BDB5B98BED_ed.png",0)
    mask_end_systole  = cv2.imread("C:/Users/Sheila/Desktop/DLUT/Medical_Seg/DataSets/EchoNet-Dynamic/mask/0X1A2A76BDB5B98BED_es.png",0)
    kp_data = pd.read_csv("C:/Users/Sheila/Desktop/KeyPoints.csv")
    aimdata = kp_data.loc[kp_data['imagename'] == '0X1A2A76BDB5B98BED_ED .png', ['p1_x', 'p1_y','p2_x', 'p2_y','p3_x', 'p3_y']]
    aim_data = np.array(aimdata).tolist()[0]
    calculate_V(mask_end_diastole,aimdata)



