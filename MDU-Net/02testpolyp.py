import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from UARNet.UCRNetAB1 import UCRNet10_2
from utils.dataloader import test_dataset
import imageio
import sklearn.metrics as metrics
import sys


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
    iou = TP / (TP + FN + FP)
    dice = (2*TP) / ((TP+FN)+(TP+FP))
    acc = (TP + TN) / (TP + FN + TN + FP)
    sen = TP / (TP + FN)
    sp = TN / (TN + FP)
    pre = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2*((pre*recall)/(pre + recall))
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
                        default='polyp/kUCRNet10_2/UCRNet10_2.pth')  # 'snapshots/TransFuse-19_best.pth'
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--save_path', type=str, default=None, help='path to save inference segmentation')#'result/Trans_unet2/isic'
    #'polyp4/result/test5'

    opt = parser.parse_args()

    model = UCRNet10_2().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))
    model.cuda()
    model.eval()

    if opt.save_path is not None:
        os.makedirs(opt.save_path, exist_ok=True)
    print('evaluating model: ', opt.ckpt_path)
    #for s in ['test1', 'test2', 'test3', 'test4', 'test5']:
    for s in ['test1', 'test2']:
        image_root = '{}/data_{}.npy'.format(opt.test_path, s)
        gt_root = '{}/mask_{}.npy'.format(opt.test_path, s)
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
            # print(test_loader.size)
            image, gt = test_loader.load_data()  #
            gt = 1 * (gt > 0.5)
            image = image.cuda()

            with torch.no_grad():
                #_, _, _, _, res = model(image)
                # res2, res3,res4, res = model(image)
                res = model(image)
            # res = res1+res2+res3+res4+res
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = 1 * (res > 0.5)

            if opt.save_path is not None:
                imageio.imwrite(opt.save_path + '/' + str(i) + '_pred.jpg', res)
                imageio.imwrite(opt.save_path + '/' + str(i) + '_gt.jpg', gt)
            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)
            acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])
            # iou1, dice1, acc1, sen, sp, pre, recall, f1 = accuracy(res, gt)
            # total_auc.append(calculate_auc_test(res, gt))

            acc_bank.append(acc)
            dice_bank.append(dice)
            iou_bank.append(iou)

        sys.stdout = open("kpolypUCRNet10_2.txt", "a+")
        print("###################")
        print("Data", s)
        print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
              format(np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
        print("********************************************************")
