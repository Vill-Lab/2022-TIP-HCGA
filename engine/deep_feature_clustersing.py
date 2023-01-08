import imageio

import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from utils.iotools import mkdir_if_missing


class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x1 = x[1:, :, :, :]
        x2 = x[0:1, :, :, :]

        h_feature_orign = h_feature

        low_feature = self.down_l(x1)
        h_feature = self.down_h(x2)

        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm  # 扭曲坐标

        output = F.grid_sample(input, grid)  # 双线性插值得到高分辨率特征
        return output

def split_n(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def clustering(cfg, tensor, p_id):
    if p_id <= 10:
        VISULIZE = True
    else:
        VISULIZE = False

    tensor = torch.tensor(tensor).cuda()
    if cfg.DATASETS.NAMES == 'occluded_reid':
        tensor = tensor[[0, 2, 4, 6, 8, 1, 3, 5, 7, 9]]

    model = MyNet(inp_dim=tensor.shape[1], mod_dim1=cfg.DFC.MODIM1, mod_dim2=cfg.DFC.MODIM2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.03, momentum=0.0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.3, weight_decay=0.0005)

    label_colours = np.random.randint(255, size=(100, 3))
    human_colors = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [190, 113, 27], [242, 166, 217], [146, 114, 209],
         [242, 209, 31]])
    n_part = cfg.CLUSTERING.PART_NUM - 1
    spatial = cfg.DFC.SPATIAL
    show = 0

    '''train init'''
    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # pixel-level and image-level contrastive loss
    if cfg.DFC.C_LOSS == 'l1':
        loss_h = torch.nn.L1Loss(reduction='mean')  # 之前为reduction='mean'
        loss_v = torch.nn.L1Loss(reduction='mean')
        loss_diag1 = torch.nn.L1Loss(reduction='mean')
        loss_diag2 = torch.nn.L1Loss(reduction='mean')
        loss_image = torch.nn.L1Loss(reduction='mean')

    if cfg.DFC.C_LOSS == 'l2':
        loss_h = torch.nn.MSELoss(reduction='mean')
        loss_v = torch.nn.MSELoss(reduction='mean')
        loss_diag1 = torch.nn.MSELoss(reduction='mean')
        loss_diag2 = torch.nn.MSELoss(reduction='mean')
        loss_image = torch.nn.MSELoss(reduction='mean')

    h_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2] - 1, tensor.shape[3]).cuda()
    v_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2], tensor.shape[3] - 1).cuda()
    diag_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2]-1, tensor.shape[3]-1).cuda()
    img_target = torch.zeros(tensor.shape[0]-1, cfg.DFC.MODIM2, tensor.shape[2], tensor.shape[3]).cuda()

    '''train loop'''
    model.train()
    frames = []
    num = tensor.shape[0]
    h = tensor.shape[2]
    w = tensor.shape[3]

    for batch_idx in range(cfg.DFC.EPOCH):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)

        # computer pixel-level contrastive loss
        contra_h = output[:, :, spatial:, :] - output[:, :, 0:-spatial, :]
        contra_v = output[:, :, :, spatial:] - output[:, :, :, 0:-spatial]
        contra_diag1 = output[:, :, 1:, 1:]-output[:, :, 0:-1, 0:-1]
        contra_diag2 = output[:, :, 1:, 0:-1] - output[:, :, 0:-1, 1:]
        
        if cfg.DFC.RXR == 5:
            spatial2 = 2
            contra_h2 = output[:, :, spatial2:, :] - output[:, :, 0:-spatial2, :]
            contra_v2 = output[:, :, :, spatial2:] - output[:, :, :, 0:-spatial2]
            contra_diag12 = output[:, :, 1:, 1:]-output[:, :, 0:-1, 0:-1]
            contra_diag22 = output[:, :, 1:, 0:-1] - output[:, :, 0:-1, 1:]


        # computer image-level contrastive loss
        contra_img = output[1:, :, :, :] - output[0:-1:, :, :, :]

        if cfg.DFC.C_LOSS == 'cc':
            l_h = cc_loss(contra_h, h_target, loss_fn)
            l_v = cc_loss(contra_v, v_target, loss_fn)
            l_diag1 = cc_loss(contra_diag1, diag_target, loss_fn)
            l_diag2 = cc_loss(contra_diag2, diag_target, loss_fn)
            l_img = cc_loss(contra_img, img_target, loss_fn)
        else:
            l_h = loss_h(contra_h, h_target)
            l_v = loss_v(contra_v, v_target)
            l_diag1 = loss_diag1(contra_diag1, diag_target)
            l_diag2 = loss_diag2(contra_diag2, diag_target)
            l_img = loss_image(contra_img, img_target)

            if cfg.DFC.RXR == 5:
                l_h += loss_h(contra_h2, torch.zeros_like(contra_h2).cuda())
                l_v += loss_v(contra_v2, torch.zeros_like(contra_v2).cuda())
                l_diag1 = loss_diag1(contra_diag12, torch.zeros_like(contra_diag12).cuda())
                l_diag2 = loss_diag2(contra_diag22, torch.zeros_like(contra_diag22).cuda())

        output = output.permute(0, 2, 3, 1).view(output.shape[0], -1, cfg.DFC.MODIM2)
        output = output.reshape(-1, cfg.DFC.MODIM2)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()

        '''backward'''
        output_1d = output.reshape([-1, cfg.DFC.MODIM2])
        target_1d = target.reshape([-1])
        loss = loss_fn(output_1d, target_1d) + cfg.DFC.IMAGE_WEIGHT * l_img \
                + cfg.DFC.PIXEL_WEIGHT * (l_h + l_v) + cfg.DFC.DIAG_WEIGHT * (l_diag1 + l_diag2)

        loss.backward()
        optimizer.step()

        '''show image'''
        un_label, hist = np.unique(im_target, return_counts=True)
        if VISULIZE:
            im_target = im_target.reshape(tensor.shape[0], tensor.shape[2], tensor.shape[3])
            p_image = np.concatenate([im_target[i] for i in range(im_target.shape[0])], axis=1)
            p_image_rgb = np.array([label_colours[c % 100] for c in p_image])
            p_image_rgb = p_image_rgb.astype(np.uint8)

            # label post-processing
            im_target = im_target.reshape(num, h, w)

            # 根据边界除去背景
            b_labels = []
            top_bottom_boundary = np.hstack(
                    (im_target[:, 0, :], im_target[:, 3, :], im_target[:, h - 1, :], im_target[:, h - 4, :]))
            boundary_labels, hist = np.unique(top_bottom_boundary, return_counts=True)
            # print(boundary_labels, hist)
            for i in range(len(hist)):
                if hist[i] > num * w / 2:
                    b_labels.append(boundary_labels[i])

            left_right_boundary = np.hstack((im_target[:, :, 0], im_target[:, :, w - 1]))
            boundary_labels, hist = np.unique(left_right_boundary, return_counts=True)
            for i in range(len(hist)):
                if hist[i] > num * h / 3:
                    b_labels.append(boundary_labels[i])
            for i in range(len(b_labels)):
                im_target = np.where(im_target == b_labels[i], 100, im_target)

            # 在前景上根据label的平均高度分割人体
            f_labels = list(set(un_label)-set(b_labels))
            # print(b_labels, f_labels)
            mean_h = []
            for i in range(len(f_labels)):
                pos = np.where(im_target == f_labels[i])
                mean_h.append(np.mean(pos[1]))

            # 均分法
            if cfg.DFC.LR == 'mean':
                sorted_mean_h = np.argsort(mean_h)
                human_part = split_n(sorted_mean_h, n_part)
                for i in range(n_part):
                    for j in range(len(human_part[i])):
                        im_target = np.where(im_target == f_labels[human_part[i][j]], 101 + i, im_target)

            # 结构分法
            if cfg.DFC.LR =='structure':
                h_seg = (max(mean_h)-min(mean_h))/n_part
                head_labels = []
                body_labels = []
                leg_labels = []
                for i in range(len(f_labels)):
                    if mean_h[i] >= min(mean_h) and mean_h[i] < (min(mean_h) + h_seg / 2):
                        head_labels.append(f_labels[i])
                    if mean_h[i] >= (min(mean_h) + h_seg / 2) and mean_h[i] < (min(mean_h) + 2 * h_seg):
                        body_labels.append(f_labels[i])
                    if mean_h[i] >= (min(mean_h) + 2 * h_seg):
                        leg_labels.append(f_labels[i])
                # print(head_labels, body_labels, leg_labels)
                for i in range(len(head_labels)):
                    im_target = np.where(im_target == head_labels[i], 101, im_target)
                for i in range(len(body_labels)):
                    im_target = np.where(im_target == body_labels[i], 102, im_target)
                for i in range(len(leg_labels)):
                    im_target = np.where(im_target == leg_labels[i], 103, im_target)

            human_rgb = np.array([human_colors[(c-100) % (n_part + 1)] for c in im_target])
            human_rgb = np.concatenate([human_rgb[i] for i in range(im_target.shape[0])], axis=1)
            human_rgb = human_rgb.astype(np.uint8)

            show = np.vstack((p_image_rgb, human_rgb))

            if (batch_idx+1) % 2 == 0:
                frames.append(show)
                # cv2.imwrite("cl/temp/batch=%s.png" % batch_idx, show)
        if len(un_label) <= cfg.DFC.MIN_LABEL_NUM:
            break
        # if (l_h + l_v).item() < 0.2:
        #     break

    '''save'''
    if VISULIZE:
        save_dir = 'cw=%s_tw=%s_dw=%s_gpu=%s' % (
        cfg.DFC.PIXEL_WEIGHT, cfg.DFC.IMAGE_WEIGHT, cfg.DFC.DIAG_WEIGHT, cfg.MODEL.DEVICE_ID)
        mkdir_if_missing(save_dir)
        imageio.imsave("%s/label-num=%s_id=%s.png" % (save_dir, cfg.DFC.MODIM2, p_id), show)
        # show = show.transpose(1, 2, 0)
        # imageio.mimsave("%s/label-num=%s_id=%s.gif" % (save_dir, cfg.DFC.MODIM2, p_id), frames, 'GIF', duration=0.1)

    # label post-processing
    if VISULIZE == False:
        im_target = im_target.reshape(num, h, w)

        # 根据边界除去背景
        # b_labels = list(un_label[np.argsort(hist)[-1:]])
        b_labels = []
        top_bottom_boundary = np.hstack(
            (im_target[:, 0, :], im_target[:, 3, :], im_target[:, h - 1, :], im_target[:, h - 4, :]))
        boundary_labels, hist = np.unique(top_bottom_boundary, return_counts=True)
        # print(boundary_labels, hist)
        for i in range(len(hist)):
            if hist[i] > num * w / 2:
                b_labels.append(boundary_labels[i])
        left_right_boundary = np.hstack((im_target[:, :, 0], im_target[:, :, w - 1]))
        boundary_labels, hist = np.unique(left_right_boundary, return_counts=True)
        for i in range(len(hist)):
            if hist[i] > num * h / 3:
                b_labels.append(boundary_labels[i])
        for i in range(len(b_labels)):
            im_target = np.where(im_target == b_labels[i], 100, im_target)

        # 在前景上根据label的平均高度分割人体
        f_labels = list(set(un_label) - set(b_labels))
        # print(b_labels, f_labels)
        mean_h = []
        for i in range(len(f_labels)):
            pos = np.where(im_target == f_labels[i])
            mean_h.append(np.mean(pos[1]))

        # 均分法
        if cfg.DFC.LR == 'mean':
            sorted_mean_h = np.argsort(mean_h)
            human_part = split_n(sorted_mean_h, n_part)
            for i in range(n_part):
                for j in range(len(human_part[i])):
                    im_target = np.where(im_target == f_labels[human_part[i][j]], 101 + i, im_target)

        # 结构分法
        if cfg.DFC.LR == 'structure':
            h_seg = (max(mean_h) - min(mean_h)) / n_part
            head_labels = []
            body_labels = []
            leg_labels = []
            for i in range(len(f_labels)):
                if mean_h[i] >= min(mean_h) and mean_h[i] < (min(mean_h) + h_seg / 2):
                    head_labels.append(f_labels[i])
                if mean_h[i] >= (min(mean_h) + h_seg / 2) and mean_h[i] < (min(mean_h) + 2 * h_seg):
                    body_labels.append(f_labels[i])
                if mean_h[i] >= (min(mean_h) + 2 * h_seg):
                    leg_labels.append(f_labels[i])
            # print(head_labels, body_labels, leg_labels)
            for i in range(len(head_labels)):
                im_target = np.where(im_target == head_labels[i], 101, im_target)
            for i in range(len(body_labels)):
                im_target = np.where(im_target == body_labels[i], 102, im_target)
            for i in range(len(leg_labels)):
                im_target = np.where(im_target == leg_labels[i], 103, im_target)

    return im_target, show

