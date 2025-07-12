import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from UARNet.UARNet8 import UARNet10
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
# 进行可视化
#from visdom import Visdom
import numpy as np
# import time


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    # return (wbce + wiou).mean()
    return (wbce).mean


def train(train_loader, model, optimizer, epoch, best_loss):
    model.train()
    # ---- multi-scale training ----
    #size_rates = [0.75, 1, 1.25]
    size_rates = [0.75, 1, 1.25]
    #loss_record2, loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter()
    loss_record2 = AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize1 = int(round(800 * rate / 32) * 32)
            trainsize2 = int(round(800 * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize1, trainsize2), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize1, trainsize2), mode='bilinear', align_corners=True)
            # ---- forward ----
            # lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
            lateral_map_2 = model(images)  # map_3=map_x, map_2=map_1, map_4map_2
            # ---- loss function ----
            # if rate == 1:
            # loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            # loss4 = structure_loss(lateral_map_4, gts)
            # loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4
            loss = loss2
            # ---- backward ----
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                # loss_record3.update(loss3.data, opt.batchsize)
                # loss_record4.update(loss4.data, opt.batchsize)
            # ---- train visualization ----
            if i % 20 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[lateral-2: {:.4f}]'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             loss_record2.show()))

    save_path = 'zhijia/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 1 == 0:
        meanloss = test(model, opt.test_path)
        # 更新窗口图像
        #viz.line([[best_loss, meanloss]], [epoch], win='train', update='append')
        # 延时0.5s
        #time.sleep(0.5)
        if meanloss < best_loss:
            print('new best loss: ', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), save_path + 'UARNet10.pth')
            print('[Saving Snapshot:]', save_path + 'UARNet10-%d.pth' % epoch)
    """if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), save_path + 'UARNet1-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'UARNet1-%d.pth' % epoch)"""
    return best_loss


def test(model, path):
    model.eval()
    mean_loss = []

    # for s in ['val', 'test']:#原来的
    for s in ['val']:
        image_root = '{}/data_{}.npy'.format(path, s)
        gt_root = '{}/mask_{}.npy'.format(path, s)
        test_loader = test_dataset(image_root, gt_root)
        print("val-length", test_loader.size)

        dice_bank = []
        iou_bank = []
        loss_bank = []
        acc_bank = []

        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.cuda()

            with torch.no_grad():
                #_, _, res = model(image)
                res = model(image)
            loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

            res = res.sigmoid().data.cpu().numpy().squeeze()
            gt = 1 * (gt > 0.5)
            res = 1 * (res > 0.5)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)
            acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            acc_bank.append(acc)

        print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
              format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))
        # 更新窗口图像
        #viz.line([[np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)]], [epoch], win='val', update='append')
        # 延时0.5s
        #time.sleep(0.5)
        mean_loss.append(np.mean(loss_bank))

    return mean_loss[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='UARNet10Loss')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()

    # ---- build models ----
    model = UARNet10().cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))

    image_root = '{}/data_train.npy'.format(opt.train_path)
    gt_root = '{}/mask_train.npy'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    print("train-length", len(train_loader))
    total_step = len(train_loader)
    # 将窗口类实例化
    """viz = Visdom()
    # 创建窗口并初始化
    viz.line([[0.0, 0.0]], [0], win='train', opts=dict(title='bestloss&meanloss', legend=['bestloss', 'meanloss']))
    viz.line([[0.0, 0.0, 0.0]], [0], win='val', opts=dict(title='dice&iou&acc', legend=['dice', 'iou', 'acc']))
    # viz.line([[0.0, 0.0, 0.0]], [0], win='trainloss', opts=dict(title='loss2&loss2&loss4', legend=['loss2', 'loss3', 'loss4']))"""
    #
    print("#" * 20, "Start Training", "#" * 20)

    # best_loss = 1e5
    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss)
