from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse

from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
import logging
# 配置日志记录
log_file = 'training_loss.log'  #
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
def train(args, opt, train_mode ="all"):
    val_preloss = 100
    list_fold = list(range(0, args.k_fold + 1))
    model = get_model(args.modelname, args=args, opt=opt)
    for name, param in model.named_parameters():
        if "image_encoder.pos_embed" in name or "image_encoder.patch_embed.proj" in name:  # 根据名称判断是否为需要冻结的层   目前 bert True
            param.requires_grad = True
        if "image_encoder" and "layer_ff" in name:
            param.requires_grad = True
        # if "image_encoder" and "mlp" in name:
        #     param.requires_grad = True
        # if "image_encoder" and "norm2" in name:
        #     param.requires_grad = True
        # if "image_encoder" and "ReMLP" in name:
        #     param.requires_grad = True
        # if "image_encoder" and "norm3" in name:
        #     param.requires_grad = True
        if 'bert' in name:  # 根据名称判断是否为需要冻结的层
            param.requires_grad = False
        if 'mask_decoder' in name:  # 根据名称判断是否为需要冻结的层
            param.requires_grad = False
    fold_list_train = list_fold.copy()
    fold_list_val = [args.k_fold]
    fold_list_train.remove(args.k_fold)
    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None,
                                long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size,
                              crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(fold_list_train, opt.data_path, args.datasets, tf_train,
                                   img_size=args.encoder_input_size, train_model=train_mode,train_flag = True)
    val_dataset = ImageToImage2D(fold_list_val, opt.data_path, args.datasets, tf_val,
                                 img_size=args.encoder_input_size,
                                 train_model=train_mode,train_flag = False)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model.to(device)

    # print params
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    for name, param in model.named_parameters():  # 打印参数训练情况
        print(f"{name}: {param.requires_grad}")
    if args.modelname == "EchoSAM_KP":
        optimizer1 = optim.Adam([{'params': model.image_encoder.parameters()},
                                 {'params': model.prompt_encoder.parameters()},
                                 {'params': model.mask_decoder.parameters()}, ],
                                lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer2 = optim.Adam([{'params': model.KeyPointDecoder.parameters()}, ],
                                lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion_kp = nn.MSELoss()
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                               amsgrad=False)

    criterion = get_criterion(modelname=args.modelname, opt=opt)
    iter_num = 0
    max_iterations = args.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(args.epochs + 1), np.zeros(args.epochs + 1)
    seg_stop_flag = 100

    for epoch in range(args.epochs):  # 遍历每一轮
        if epoch == seg_stop_flag:
            for name, param in model.named_parameters():
                if "prompt_encoder" in name:  # 根据名称判断是否为需要冻结的层   目前 bert True
                    param.requires_grad = False
                if "image_encoder" in name:
                    param.requires_grad = False
                if "mask_decoder" in name:
                    param.requires_grad = False
                if 'bert' in name:
                    param.requires_grad = False
        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['label'].to(dtype=torch.float32, device=opt.device)
            text_prompt = datapack['text_prompt']
            pt = get_click_prompt(datapack, opt)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            keypoints = datapack['keypoints'].to(opt.device)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs,text_prompt,None) # img,text,None,pt/box
            # pred = model(imgs, None, bbox)  # img,pt,box-----SAM
            # -------------------------------------------------------- backward -------------------------------------------------------
            if args.modelname == "EchoSAM_KP":
                train_loss_1 = criterion(pred, masks)
                train_loss_2 = criterion_kp(pred["keypoints"], keypoints)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                train_loss_1.backward(retain_graph=True)
                train_loss_2.backward()
                optimizer1.step()  # update the parameters
                optimizer2.step()
                train_loss = train_loss_1 + train_loss_2
            else:
                train_loss = criterion(pred, masks)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            print(train_loss)
            train_losses = train_losses + train_loss
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (
                                1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer1.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer1.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)
        model.eval()
        val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
        logging.info(f'Epoch {epoch}: Test Loss = {val_losses:.4f}')
        print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, opt.epochs, val_losses))
        if val_losses < val_preloss:
            val_preloss = val_losses
            save_path = "./checkpoints/best_" + str(k_fold)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)

        # Save the model file every ten rounds
        if epoch == 0 or epoch == 49 or epoch == 99 or epoch == 89 or epoch == 79:
            if not os.path.isdir(opt.save_path):
                os.makedirs(opt.save_path)
            save_path = "./checkpoints/best_" + str(epoch) + "_" + str(k_fold)
            torch.save(model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='EchoSAM_KP', type=str, help='type of model, e.g., SAM, EchoSAM...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='US30K', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/saved_checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/SAMUS_04091843_125_0.07263953642606145.pth', help='Pretrained checkpoint of SAM')
    # parser.add_argument('--sam_ckpt', type=str, default='./checkpoints/Fissures_crack500_100e.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--datasets', type=str, default="EchoNet-Dynamic", help='EchoNet-Dynamic or CAMUS')
    parser.add_argument('--k_fold', type=int, default=1, help='K-fold cross-validation')
    parser.add_argument('--base_lr', type=float, default=0.00005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--epochs', type=int, default=100, help='') #
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    args = parser.parse_args()
    opt = get_config(args.task)
    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution
    # CAMUS 的train_mode = "four" or "two"
    # EchoNet-Dynamic train_mode = "all"
    for k_fold in range(args.k_fold):
        train(args, opt,train_mode="all")







