import os
from random import randint
import numpy as np
import torch
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from einops import rearrange
import random

import math


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# import csv
# with open('KeyPoints.csv', mode='w', newline='') as file:  # 单独运行
#     writer = csv.writer(file)
#     writer.writerow(["imagename", "class", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y"])


def get_one_mask(mask):
    # mask = cv2.imread(mask_path, 0)
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
def get_keypoints(imagepath,mask):
    df = pd.read_csv('KeyPoints.csv')
    basename =  imagepath.split('/')[-1]
    mask_1, mask_2, mask_3 = get_one_mask(mask)
    edges_1 = cv2.Canny(mask_1, 100, 200)
    contours_1,_ = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # myo


    edges_2 = cv2.Canny(mask_2, 100, 200)
    contours_2,_ = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # lv

    edges_3 = cv2.Canny(mask_3, 100, 200)
    contours_3,_ = cv2.findContours(edges_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # la

    # cv2.drawContours(mask_1, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("Image with Contours", mask_1)
    edges_points_lv = get_edges_points(contours_2)
    edges_points_la = get_edges_points(contours_3)
    edges_points_myo = get_edges_points(contours_1)
    same = [0,0,0,0,0,0]
    mindis = 1000000  # 保存最小
    same1 = [0,0,0,0,0,0]
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
                if mindis3 < mindis:  # mindis3是最小的点   and same1[0]-same[0] > 10
                    mindis2 = mindis
                    same2 = same  # 更新第二小值
                    mindis = mindis3  # 更新最小值
                    same = [lv_x, lv_y, la_x, la_y, myo_x, myo_y]
                    if abs(lv_x - same2[0]) > 5:
                        mindis1 = mindis2
                        same1 = same2  # 更新第二小值
                    # else:
                    #     mindis = mindis3  #  更新最小值
                    #     same = [lv_x,lv_y,la_x,la_y,myo_x,myo_y]
                    # same.append((lv_x,lv_y,la_x,la_y,myo_x,myo_y))
    p3_y = min(lv_y_l)
    index_ = lv_y_l.index(p3_y)
    p3_x = lv_x_l[index_]

    print(same)
    print(same1)
    try:
        p3_y_la = max(la_y_l)
        index_la = la_y_l.index(p3_y_la)
        p3_x_la = la_x_l[index_la]
    except:
        print("!!!:",la_y_l)
        p3_y_la = 0
        p3_x_la = 0

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

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]

def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt = indices[len(indices)//2] 
    return pt[np.newaxis, :], [point_label]


def random_clicks(mask, class_id = 1, prompts_number=10):
    indices = np.argwhere(mask == class_id)
    indices[:, [0,1]] = indices[:, [1,0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0,1]] = indices[:, [1,0]]
    pt_index = np.random.randint(len(indices), size=prompts_number)
    pt = indices[pt_index]
    point_label = np.repeat(point_label, prompts_number)
    return pt, point_label

def pos_neg_clicks(mask, class_id=1, pos_prompt_number=5, neg_prompt_number=5):
    pos_indices = np.argwhere(mask == class_id)
    pos_indices[:, [0,1]] = pos_indices[:, [1,0]]
    pos_prompt_indices = np.random.randint(len(pos_indices), size=pos_prompt_number)
    pos_prompt = pos_indices[pos_prompt_indices]
    pos_label = np.repeat(1, pos_prompt_number)

    neg_indices = np.argwhere(mask != class_id)
    neg_indices[:, [0,1]] = neg_indices[:, [1,0]]
    neg_prompt_indices = np.random.randint(len(neg_indices), size=neg_prompt_number)
    neg_prompt = neg_indices[neg_prompt_indices]
    neg_label = np.repeat(0, neg_prompt_number)

    pt = np.vstack((pos_prompt, neg_prompt))
    point_label = np.hstack((pos_label, neg_label))
    return pt, point_label

def random_bbox(mask, class_id=1, img_size=256):
    # return box = np.array([x1, y1, x2, y2])
    # mask_ = np.where(mask == class_id, 1, 0)
    # mask_1 = np.where(mask_ == 1, 255, mask_)
    class_id = 1
    indices = np.argwhere(mask == int(class_id)) # Y X
    indices[:, [0,1]] = indices[:, [1,0]] # x, y
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx-minx+1
    classh_size = maxy-miny+1

    shiftw = randint(int(0.95*classw_size), int(1.05*classw_size))
    shifth = randint(int(0.95*classh_size), int(1.05*classh_size))
    shiftx = randint(-int(0.05*classw_size), int(0.05*classw_size))
    shifty = randint(-int(0.05*classh_size), int(0.05*classh_size))

    new_centerx = (minx + maxx)//2 + shiftx
    new_centery = (miny + maxy)//2 + shifty

    minx = np.max([new_centerx-shiftw//2, 0])
    maxx = np.min([new_centerx+shiftw//2, img_size-1])
    miny = np.max([new_centery-shifth//2, 0])
    maxy = np.min([new_centery+shifth//2, img_size-1])

    return np.array([minx, miny, maxx, maxy])

def fixed_bbox(mask, class_id = 1, img_size=256):
    indices = np.argwhere(mask == class_id) # Y X (0, 1)
    indices[:, [0,1]] = indices[:, [1,0]]
    if indices.shape[0] ==0:
        return np.array([-1, -1, img_size, img_size])
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy])

class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                 p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                 long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)
        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))
        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)
        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
        # transforming to tensor
        # image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        image, mask_ = F.resize(image, (256, 256), InterpolationMode.BICUBIC), F.resize(mask, (256, 256), InterpolationMode.BICUBIC)
        mask = F.resize(mask, (256, 256), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask)
            mask_ = F.to_tensor(mask_)
            low_mask = F.to_tensor(low_mask)
        else:
            mask = to_long_tensor(mask)
            mask_ = to_long_tensor(mask_)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask, mask_

CLSSS_ = [85,170,255]
class ImageToImage2D(Dataset):

    def __init__(self, fold_list,dataset_path: str,split='train', joint_transform: Callable = None, img_size=256, prompt = "click", class_id=1,
                 one_hot_mask: int = False,train_model = "all",train_flag = True) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        self.KPcsv = pd.read_csv("./datasets/EchoNet-Dynamic/KeyPoints.csv")
        self.fold_list = fold_list
        id_list_file = os.path.join(dataset_path, 'MainPatient/{0}.txt'.format(split))
        self.ids_ = [id_.strip() for id_ in open(id_list_file)]
        self.ids = []
        for element in self.ids_:
            try:
                fold_id = int(element.split("__")[0])
                if fold_id in fold_list:
                    self.ids.append(element)
            except:
                self.ids = self.ids_
        if train_model == "two":
            self.ids = [elemet for elemet in self.ids if "2CH" in elemet]
            # element for element in my_list if target_string in element
        elif train_model == "four":
            self.ids = [elemet for elemet in self.ids if "4CH" in elemet]
        else:
            pass

        self.prompt = prompt
        self.img_size = img_size
        self.class_id = class_id
        self.class_dict_file = os.path.join(dataset_path, 'MainPatient/class.json')
        with open(self.class_dict_file, 'r') as load_f:
            self.class_dict = json.load(load_f)
        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
        # self.ids = self.ids_
        if self.split == "EchoNet-Dynamic":
            self.ids = self.ids_
            if train_flag == True:
                for imgname in self.ids:
                    if "test" in imgname:
                        self.ids = list(filter(lambda x: x != imgname, self.ids))
            else:
                for imgname in self.ids:
                    if "train" in imgname:
                        self.ids = list(filter(lambda x: x != imgname, self.ids))
            for imgname in self.ids:
                imgname_ = imgname.split("/")[-1]
                root_img = "./datasets/EchoNet-Dynamic/img/"
                root_label = "./datasets/EchoNet-Dynamic/label/"
                image = cv2.imread(root_img+imgname_+".png", 0)
                mask = cv2.imread(root_label+imgname_+".png", 0)
                if image is None or mask is None:
                    self.ids = list(filter(lambda x: x != imgname, self.ids))
        if self.split == "EchoDUT":
            self.ids = self.ids_
            if train_flag == True:
                for imgname in self.ids:
                    if "test" in imgname:
                        self.ids = list(filter(lambda x: x != imgname, self.ids))
            else:
                for imgname in self.ids:
                    if "train" in imgname:
                        self.ids = list(filter(lambda x: x != imgname, self.ids))
            for imgname in self.ids:
                imgname_ = imgname.split("/")[-1]
                root_img = "./datasets/EchoDUT/img/"
                root_label = "./datasets/EchoDUT/label/"
                image = cv2.imread(root_img+imgname_+".png", 0)
                mask = cv2.imread(root_label+imgname_+".png", 0)
                if image is None or mask is None:
                    self.ids = list(filter(lambda x: x != imgname, self.ids))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        if "EchoNet-Dynamic" in id_:  # EchoNet-Dynamic数据集
            text_ = ["extract ", "segment ", "please divided "]
            main_classes = {"255":"Left ventricle"}
            if "test" in self.split:
                class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[2], id_.split('/')[3]
            else:
                class_id0_fold_id, sub_path, filename = id_.split('/')[0], id_.split('/')[2], id_.split('/')[3]
            img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
            label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
            image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
            mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)

            # pt, point_label = random_clicks(np.array(mask), class_id=class_id, prompts_number=5)


            image, mask = correct_dims(image, mask)
            # image = image.sequeeze(0)
            if self.joint_transform:
                image, mask, low_mask, mask_ = self.joint_transform(image, mask)
                # image = image.squeeze(0)
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)


            kp_row = self.KPcsv[(self.KPcsv["imagename"] == filename + ".png")]  #  & (self.KPcsv["class"] == kp_classes[str(class_id0)])]
            kp_row_aim = kp_row[["p1_x","p1_y","p2_x","p2_y","p3_x","p3_y"]]
            kp_tensor = torch.tensor(kp_row_aim.values/112, dtype=torch.float32)  # 地标归一化
            kp_tensor = kp_tensor.squeeze(0)

            pre_text = random.choice(text_)
            classes = main_classes[str(255)]
            text_prompt = {"caption": pre_text + classes}
            class_id = 255

            mask[mask != class_id] = 0
            mask[mask == class_id] = 1
            low_mask[low_mask != class_id] = 0
            low_mask[low_mask == class_id] = 1
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'veone_hot_mask must be nonnegati'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

            low_mask = low_mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            # pt, point_label = random_click(np.array(mask), class_id)
            pt, point_label = random_clicks(np.array(mask), class_id=class_id, prompts_number=5)
            bbox = random_bbox(np.array(mask_), class_id)
            # print(1)

        elif "EchoDUT" or "Fissures" in id_: # EchoDUT数据集
            text_ = ["extract ", "segment ", "please divided "]
            main_classes = {"255":"Left ventricle"}  # Fissures文件不使用文本提示
            if "test" in self.split:
                class_id0, sub_path, filename = id_.split('/')[0], id_.split('/')[2], id_.split('/')[3]
            else:
                class_id0_fold_id, sub_path, filename = id_.split('/')[0], id_.split('/')[2], id_.split('/')[3]
            img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
            label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
            image = cv2.imread(os.path.join(img_path, filename + '.jpg'), 0)
            mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)

            image, mask = correct_dims(image, mask)
            # image = image.sequeeze(0)
            if self.joint_transform:
                image, mask, low_mask, mask_ = self.joint_transform(image, mask)
                # image = image.squeeze(0)
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

            pre_text = random.choice(text_)
            classes = main_classes[str(255)]
            text_prompt = {"caption": pre_text + classes}
            class_id = 255

            mask[mask != class_id] = 0
            mask[mask == class_id] = 1

            mask_[mask_ != class_id] = 0
            mask_[mask_ == class_id] = 1

            low_mask[low_mask != class_id] = 0
            low_mask[low_mask == class_id] = 1
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'veone_hot_mask must be nonnegati'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

            pt, point_label = random_clicks(np.array(mask), class_id=class_id, prompts_number=5)
            if "Fissures" in self.split:
                bbox = random_bbox(np.array(mask_), class_id, 256)
            else:
                bbox = random_bbox(np.array(mask), class_id, 256)
            low_mask = low_mask.unsqueeze(0)
            mask = mask_.unsqueeze(0)
            kp_row_aim = [0,0,0,0,0,0]
            kp_tensor = torch.tensor(kp_row_aim, dtype=torch.float32)


        else:

            text_ = ["extract ", "segment ", "please divided "]
            main_classes = {"170": "ventricular wall", "85": "Left ventricle", "255": "left atrium"}
            kp_classes = {"170": "myo", "85": "lv", "255": "la"}
            class_id0_fold_id, sub_path, filename = id_.split('/')[0], id_.split('/')[1], id_.split('/')[2]
            img_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'img')
            label_path = os.path.join(os.path.join(self.dataset_path, sub_path), 'label')
            image = cv2.imread(os.path.join(img_path, filename + '.png'), 0)
            mask = cv2.imread(os.path.join(label_path, filename + '.png'), 0)
            fold_id = class_id0_fold_id.split("__")[0]
            class_id0 = class_id0_fold_id.split("__")[1]
            pre_text = random.choice(text_)
            classes = main_classes[str(class_id0)]
            image, mask = correct_dims(image, mask)
            if self.joint_transform:
                image, mask, low_mask = self.joint_transform(image, mask)
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

             # --------- make the point prompt -----------------
            if self.prompt == 'click':
                point_label = 1
                if 'train' in self.split:
                    #class_id = randint(1, classes-1)
                    class_id = int(class_id0)
                elif 'val' in self.split:
                    class_id = int(class_id0)
                else:
                    class_id = int(class_id0)
                if 'train' in self.split:
                    # pt, point_label = random_click(np.arraFy(mask), class_id)
                    pt, point_label = random_clicks(np.array(mask), class_id=class_id, prompts_number=5)
                    bbox = random_bbox(np.array(mask), class_id, self.img_size)
                else:
                    # pt, point_label = random_click(np.array(mask), class_id)
                    pt, point_label = random_clicks(np.array(mask), class_id=class_id, prompts_number=5)
                    bbox = random_bbox(np.array(mask), class_id, self.img_size)

                mask[mask!=class_id] = 0
                mask[mask==class_id] = 1
                low_mask[low_mask!=class_id] = 0
                low_mask[low_mask==class_id] = 1

                point_labels = np.array(point_label)
            if self.one_hot_mask:
                assert self.one_hot_mask > 0, 'veone_hot_mask must be nonnegati'
                mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
            # mask = mask.cpu().numpy().astype(np.uint8)
            # low_mask = low_mask.cpu().numpy().astype(np.uint8)
            low_mask = low_mask.unsqueeze(0)
            mask = mask.unsqueeze(0)
            kp_row_aim = [0,0,0,0,0,0]
            kp_tensor = torch.tensor(kp_row_aim, dtype=torch.float32)
            text_prompt = {"caption":pre_text+classes}

        return {
            'image': image,
            'label': mask,
            "keypoints": kp_tensor,
            # 'p_label': point_labels,
            'pt': pt,
            'bbox': bbox,
            'low_mask':low_mask,
            'image_name': filename + '.png',
            'class_id': class_id,
            "text_prompt":pre_text+classes
            }


class Logger:
    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


