import os
import cv2
from utils.pck import pck_cal
from utils.evaluation_metrics import dice_coeff, Iou, F1_Score, compute_hausdorff_distance
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
from utils.config import get_config
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.generate_prompts import get_click_prompt
from PIL import Image

def remove_max_min(lst):
    max_val = max(lst)
    min_val = min(lst)
    return [item for item in lst if item != max_val and item != min_val]
'''
import csv
headers = ['p1','p2','p3','pall']
with open('pck_csv.csv','w',encoding='utf8',newline='') as f :
    writer = csv.writer(f)
    writer.writerow(headers)
'''


def eval(train_mode="all",datasets = "CAMUS"):
    #  =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='EchoSAM_KP', type=str, help='type of model, e.g., SAM, EchoSAM...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/efechosam-0.5-2ch.pth', help='Pretrained checkpoint of SAM')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/fissures_echosam.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/EchoSAM_echonet__last.pth', help='Pretrained checkpoint of SAM')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/EchoSAM_echonet__last.pth', help='Pretrained checkpoint of SAM')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/saved_checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--datasets', type=str, default="EchoNet-Dynamic", help='EchoNet-Dynamic or CAMUS or EchoDUT,Fissures')
    parser.add_argument('--k_fold', type=int, default=1, help='K-fold cross-validation')
    parser.add_argument('--base_lr', type=float, default=0.00005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.mode = "val"
    opt.visual = True
    opt.modelname = args.modelname
    device = torch.device(opt.device)
    # calculate the evaluation metrics
    PCK = []
    Re_P1 = [0 for _ in range(100)]
    Re_P2 = [0 for _ in range(100)]
    Re_P3 = [0 for _ in range(100)]
    Re_All = [0 for _ in range(100)]
    DICE = []
    fold_list_val = [args.k_fold]
    HD = []
    IOU = []
    F1_SCORE = []
    val_data_flag = ["test"] # 用于十折交叉验证["1"]
    ef_imagename = []
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                              ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)


    val_dataset = ImageToImage2D(fold_list_val, opt.data_path, args.datasets, tf_val,
                                 img_size=args.encoder_input_size,
                                 train_model=train_mode,train_flag = False)  # return image, mask, and filename
    # val_dataset = ImageToImage2D(val_data_flag, opt.data_path, args.datasets, tf_val,
    #                              img_size=args.encoder_input_size,
    #                              train_model=train_mode,train_flag = False)  # return image, mask, and filename
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.eval()
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype = torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype = torch.float32, device=opt.device))
        text_prompt = datapack['text_prompt']
        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
        keypoints = datapack['keypoints'].to(opt.device)
        class_ids = datapack['class_id']
        image_name = datapack['image_name']
        print(image_name)
        with torch.no_grad():
            # pred = model(imgs, None,None,bbox) # img,text,None,pt/box----EchoSAM
            pred = model(imgs, text_prompt, None)  # img,text,None,pt/box----SAMUS
            # pred = model(imgs, None, bbox) # img,pt,box-----SAM
        predict = torch.sigmoid(pred['low_res_logits'])
        keys_pre = pred['keypoints']
        # key_pre = keys_pre[j]  # .view(2,3)
        # key_label = keypoints[j]
        pck_v,result_p1,result_p2,result_p3,result_all = pck_cal(keys_pre, keypoints)
        Re_P1 = [a_i + b_i for a_i, b_i in zip(Re_P1, result_p1)]
        Re_P2 = [a_i + b_i for a_i, b_i in zip(Re_P2, result_p2)]
        Re_P3 = [a_i + b_i for a_i, b_i in zip(Re_P3, result_p3)]
        Re_All = [a_i + b_i for a_i, b_i in zip(Re_All, result_all)]
        PCK.append(pck_v)
    #     predict = predict.detach().cpu().numpy()  # (b, c, h, w)
    #     images = imgs.detach().cpu().numpy()
    #     seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
    #     b, h, w = seg.shape
    #     for j in range(0, b):
    #         # bbox_ = bbox_list[j]
    #         key_pre = keys_pre[j] # .view(2,3)
    #         key_label = keypoints[j]
    #         pck_t = pck_cal(key_pre,key_label)
    #         pre_image_normal = predict[j]
    #         pre_image_normal = (pre_image_normal-pre_image_normal.min()) / (pre_image_normal.max()-pre_image_normal.min())*255
    #         pre_image_normal = pre_image_normal.astype(np.uint8)
    #         image_array = pre_image_normal[0]
    #         image_ = Image.fromarray(image_array,mode="L")
    #         image_.save("./output/pre_image_output/image.png")
    #         ef_imagename.append(datapack["image_name"][j])
    #         class_id = class_ids[j].item()
    #         image = images[j]
    #         image = np.squeeze(image)
    #         pred_i = np.zeros((1, h, w))
    #         pred_i[seg[j:j+1, :, :] == 1] = 255
    #         pred_i = pred_i[0].astype(np.uint8)
    #         pred_i[pred_i == 255] = 1
    #         pred_dice_score = torch.from_numpy(pred_i)
    #         target = masks[j].view(128, 128).cpu()
    #         target = target.numpy()
    #         hd = compute_hausdorff_distance(pred_i, target,distance="euclidean")
    #         dice = dice_coeff(pred_dice_score, masks[j],device)
    #         iou = Iou(pred_dice_score, masks[j],device)
    #         f1_score = F1_Score(pred_dice_score, masks[j],device)
    #         dice_num = round(dice.item(),2)
    #         pred_i[pred_i == 1] = 255
    #         cv2.imwrite("./output/" + str(dice_num) + "_" + image_name[j], pred_i)
    #         write_evaluation_metrics(DICE, class_id, dice.item())
    #         write_evaluation_metrics(HD, class_id, hd)
    #         write_evaluation_metrics(IOU, class_id, iou.item())
    #         write_evaluation_metrics(F1_SCORE, class_id, f1_score)
    #
    # dice_avg = get_final_means(DICE)
    # hd_avg = get_final_means(HD)
    # iou_avg = get_final_means(IOU)
    # f1_score_avg = get_final_means(F1_SCORE)
    pck = sum(PCK)/len(PCK)
    pck_1 = [num / len(PCK) for num in Re_P1]
    pck_2 = [num / len(PCK) for num in Re_P2]
    pck_3 = [num / len(PCK) for num in Re_P3]
    pck_all = [num / len(PCK) for num in Re_All]
    # print("Dice：", dice_avg)
    # print("HD：", hd_avg)
    # print("IoU：", iou_avg)
    # print("F1-Score：", f1_score_avg)
    print("PCK:" + str(pck))
    import pandas as pd
    df = pd.read_csv('pck_csv.csv')
    df["p1_"] = pck_1
    df["p2_"] = pck_2
    df["p3_"] = pck_3
    df.to_csv('pck_csv.csv', index=False)
    dice_avg, hd_avg, iou_avg, f1_score_avg = 0,0,0,0
    return dice_avg, hd_avg, iou_avg, f1_score_avg


def write_evaluation_metrics(metrics_list,class_id,value):
    target_key = class_id
    new_value = value
    for dictionary in metrics_list:
        if target_key in dictionary:
            dictionary[target_key].append(new_value)
            break
    else:
        new_dict = {target_key: [new_value]}
        metrics_list.append(new_dict)

def get_final_means(metrics_list):
    means = {}
    for dictionary in metrics_list:
        for key, values in dictionary.items():
            if values:
                mean_value = sum(values) / len(values)
                means[key] = mean_value
    return means

def write_data(train_mode,dice_255,hd_255,dice_170,hd_170,dice_85,hd_85):
    import pandas as pd
    df = pd.read_csv('data.csv')
    if train_mode == "four":
        df["4ch_la_dice"] = dice_255
        # df["4ch_la_hd"] = hd_255
        df["4ch_lv_dice"] = dice_85
        # df["4ch_lv_hd"] = hd_85
        df["4ch_myo_dice"] = dice_170
        # df["4ch_myo_hd"] = hd_170
    else:
        df["2ch_la_dice"] = dice_255
        # df["2ch_la_hd"] = hd_255
        df["2ch_lv_dice"] = dice_85
        # df["2ch_lv_hd"] = hd_85
        df["2ch_myo_dice"] = dice_170
        # df["2ch_myo_hd"] = hd_255
    df.to_csv('data.csv', index=False)


def write_data_echonet(dice_255,hd_255):
    import pandas as pd
    dice_255 = remove_max_min(dice_255)
    hd_255 = remove_max_min(hd_255)
    df = pd.read_csv('data_echonet.csv')
    df["4ch_lv_dice"] = dice_255
    # df["4ch_lv_hd"] = hd_255
    df.to_csv('data_echonet.csv', index=False)


if __name__ == '__main__':
    eval(train_mode="all",datasets = "EchoNet-Dynamic")
