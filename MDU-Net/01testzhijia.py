import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from my_net.UCRNetAB1j import UCRNet10_6
# from UARNet.UARNet8 import UARNet10
from utils.dataloader import test_dataset
import imageio
import sklearn.metrics as metrics
import sys
import cv2

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def accuracy(pred_mask, label):
    '''
    acc=(TP+TN)/(TP+FN+TN+FP)
    '''
    pred_mask = pred_mask.astype(np.uint8)
    TP, FN, TN, FP = [0, 0, 0, 0]
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i][j] == 1:
                if pred_mask[i][j] == 1:
                    TP += 1
                elif pred_mask[i][j] == 0:
                    FN += 1
            elif label[i][j] == 0:
                if pred_mask[i][j] == 1:
                    FP += 1
                elif pred_mask[i][j] == 0:
                    TN += 1
    if TP ==0:
        return None
    else:
        iou = TP / (TP + FN + FP)
        dice = (2*TP) / ((TP+FN)+(TP+FP))
        acc = (TP + TN) / (TP + FN + TN + FP)
        sen = TP / (TP + FN)
        sp = TN / (TN + FP)
        pre = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2*(pre*recall)/(pre + recall))
    return iou,dice, acc, sen,sp,pre,recall, f1
def calculate_auc_test(prediction, label):
    # read images
    # convert 2D array into 1D array
    result_1D = prediction.flatten()
    label_1D = label.flatten()


    label_1D = label_1D

    auc = metrics.roc_auc_score(label_1D, result_1D)

    # print("AUC={0:.4f}".format(auc))

    return auc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str,
                        default='zhijia/mlfc/mlfc_GCGCS.pth')   #UARNet10Loss/UARNet10.pth'# 'snapshots/TransFuse-19_best.pth'
    parser.add_argument('--test_path', type=str,
                        default='data/zhijia_/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default='testzhijiamy_dff/', help='path to save inference segmentation')#''zhijia/result2/uarn5'
    #'zhijia/result2/UARNet_cat_res_ce_esp'
    opt = parser.parse_args()

    model = UCRNet10_6().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)

    print('evaluating model: ', opt.ckpt_path)

    image_root = '{}/data_test.npy'.format(opt.test_path)
    gt_root = '{}/mask_test.npy'.format(opt.test_path)
    test_loader = test_dataset(image_root, gt_root)

    dice_bank = []
    iou_bank = []
    acc_bank = []
    total_iou = []
    total_dice = []
    total_acc = []
    total_sen = []
    total_sp = []
    total_pre = []
    total_recall = []
    total_f1 = []
    total_auc = []

    for i in range(test_loader.size):
        image, gt = test_loader.load_data()  #
        gt = 1 * (gt > 0.5)
        image = image.cuda()

        with torch.no_grad():
           # _, _, res = model(image)
           res = model(image)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 1 * (res > 0.5)
        total_auc.append(calculate_auc_test(res, gt))

        # if opt.save_path is not None:
        #     # imageio.imwrite(opt.save_path + '/' + str(i) + '_pred.png', res)
        #     # imageio.imwrite(opt.save_path + '/' + str(i) + '_gt.png', gt)
        #     cv2.imwrite(opt.save_path + '/' + str(i) + '_pred.png', res * 255)
        #     cv2.imwrite(opt.save_path + '/' + str(i) + '_gt.png', gt * 255)
        result = accuracy(res, gt)
        if result is None:
            print("tp 为 0，跳过本次计算")
            continue
        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])
        iou1, dice1, acc1, sen, sp, pre, recall, f1 = accuracy(res, gt)
        #total_auc.append(calculate_auc_test(res, gt))

        acc_bank.append(acc)
        dice_bank.append(dice)
        iou_bank.append(iou)
        total_iou.append(iou1)
        total_dice.append(dice1)
        total_acc.append(acc1)
        total_sen.append(sen)
        total_sp.append(sp)
        total_pre.append(pre)
        total_recall.append(recall)
        total_f1.append(f1)
    # sys.stdout = open("zhijiabceloss.txt", "a+")
    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
          format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
    print("##################################")
    print(np.mean(total_iou), np.std(total_iou))
    print(np.mean(total_dice), np.std(total_dice))
    print(np.mean(total_acc), np.std(total_acc))
    print(np.mean(total_sen), np.std(total_sen))
    print(np.mean(total_sp), np.std(total_sp))
    print(np.mean(total_auc), np.std(total_auc))
    print(np.mean(total_pre), np.std(total_pre))
    print(np.mean(total_recall), np.std(total_recall))
    print(np.mean(total_f1), np.std(total_f1))