import imageio

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F


def split_n(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


class ImageLoss(nn.Module):
    def __init__(self, temperature):
        super(ImageLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=1)
        self.l1 = torch.nn.L1Loss(reduction='mean')

    def mask_correlated_clusters(self, num):
        mask = torch.ones((num, num))
        mask.fill_diagonal_(0)
        mask = mask.bool()
        return mask

    def forward(self, z):
        num, dim = z.shape[0], z.shape[1]
        z_i = z[1:, :]
        z_j = z[0:-1, :]
        # sim = self.similarity_f(z_i, z_j) / self.temperature
        # print(sim.shape)
        # sim = sim.view(num-1, 1)
        # print(sim)
        # labels = torch.ones(num-1).to(sim.device).long().cuda()
        # image_loss = self.criterion(sim, labels)
        # print(image_loss)
        labels = torch.zeros((num-1, dim)).cuda()
        image_loss = self.l1((z_i-z_j), labels)
        return image_loss


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d((8, 4))
        self.projector = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

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
            nn.BatchNorm2d(mod_dim2), )

    def forward(self, x):
        pixel_vector = self.seq(x)
        image_feat = self.gap(pixel_vector)
        N = image_feat.shape[0]
        image_vector = self.projector(image_feat.reshape(N, -1))
        image_vector = normalize(image_vector, dim=1)
        return pixel_vector, image_vector


def clustering(cfg, tensor, p_id):
    # device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    # tensor = torch.from_numpy(tensor).to(device)
    model = MyNet(inp_dim=tensor.shape[1], mod_dim1=cfg.DFC.MODIM1, mod_dim2=cfg.DFC.MODIM2).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.003, momentum=0.0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.0005)

    label_colours = np.random.randint(255, size=(100, 3))
    human_colors = np.array(
        [[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [190, 113, 27], [242, 166, 217], [146, 114, 209],
         [242, 209, 31]])

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # pixel-level and image-level contrastive loss
    loss_h = torch.nn.L1Loss(reduction='mean')
    loss_v = torch.nn.L1Loss(reduction='mean')
    loss_diag1 = torch.nn.L1Loss(reduction='mean')
    loss_diag2 = torch.nn.L1Loss(reduction='mean')
    loss_image = torch.nn.L1Loss(reduction='mean')
    # loss_image = ImageLoss(temperature=1)

    h_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2] - 1, tensor.shape[3]).cuda()
    v_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2], tensor.shape[3] - 1).cuda()
    diag_target = torch.zeros(tensor.shape[0], cfg.DFC.MODIM2, tensor.shape[2] - 1, tensor.shape[3] - 1).cuda()
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
        output, image_vector = model(tensor)

        # computer pixel-level contrastive loss
        contra_h = output[:, :, 1:, :] - output[:, :, 0:-1, :]
        contra_v = output[:, :, :, 1:] - output[:, :, :, 0:-1]
        contra_diag1 = output[:, :, 1:, 1:] - output[:, :, 0:-1, 0:-1]
        contra_diag2 = output[:, :, 1:, 0:-1] - output[:, :, 0:-1, 1:]
        contra_img = output[1:, :, :, :] - output[0:-1, :, :, :]

        l_h = loss_h(contra_h, h_target)
        l_v = loss_v(contra_v, v_target)
        l_diag1 = loss_diag1(contra_diag1, diag_target)
        l_diag2 = loss_diag2(contra_diag2, diag_target)

        # computer image-level contrastive loss
        # l_img = loss_image(image_vector)
        l_img = loss_image(contra_img, img_target)

        output = output.permute(0, 2, 3, 1).view(output.shape[0], -1, cfg.DFC.MODIM2)
        output = output.reshape(-1, cfg.DFC.MODIM2)

        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()

        '''backward'''
        output_1d = output.reshape([-1, cfg.DFC.MODIM2])
        # output_1d = F.softmax(output_1d, dim=1)
        target_1d = target.reshape([-1])

        if (l_h + l_v).item() < 0.2:
            cfg.DFC.PIXEL_WEIGHT = 0.1 * cfg.DFC.PIXEL_WEIGHT
            print(cfg.DFC.PIXEL_WEIGHT)

        loss = 1 * loss_fn(output_1d, target_1d) + cfg.DFC.IMAGE_WEIGHT * l_img \
               + cfg.DFC.PIXEL_WEIGHT * (l_h + l_v) + cfg.DFC.DIAG_WEIGHT * (l_diag1 + l_diag2)
        print()
        loss.backward()
        optimizer.step()

        '''show image'''
        if cfg.DFC.VISULIZE:
            un_label, hist = np.unique(im_target, return_counts=True)
            im_target = im_target.reshape(tensor.shape[0], tensor.shape[2], tensor.shape[3])
            p_image = np.concatenate([im_target[i] for i in range(im_target.shape[0])], axis=1)
            p_image_rgb = np.array([label_colours[c % 100] for c in p_image])
            p_image_rgb = p_image_rgb.astype(np.uint8)

            # label post-processing
            num = tensor.shape[0]
            h = tensor.shape[2]
            w = tensor.shape[3]
            im_target = im_target.reshape(num, h, w)

            # 根据边界除去背景
            # b_labels = list(un_label[np.argsort(hist)[-1:]])
            b_labels = []

            top_bottom_boundary = np.hstack(
                (im_target[:, 0, :], im_target[:, 3, :], im_target[:, h - 1, :], im_target[:, h - 4, :]))
            boundary_labels, hist = np.unique(top_bottom_boundary, return_counts=True)
            for i in range(len(hist)):
                if hist[i] > num * w / 2:
                    b_labels.append(boundary_labels[i])
            # leg_b = np.hstack(im_target[:, h - 10, :])
            # boundary_labels, hist = np.unique(leg_b, return_counts=True)
            # for i in range(len(hist)):
            #     if hist[i] > num * w / 2:
            #         print(boundary_labels[i])
            #         b_labels.append(boundary_labels[i])

            left_right_boundary = np.hstack((im_target[:, :, 0], im_target[:, :, 3], im_target[:, :, w - 1]))
            boundary_labels, hist = np.unique(left_right_boundary, return_counts=True)
            for i in range(len(hist)):
                if hist[i] > num * h / 2:
                    b_labels.append(boundary_labels[i])
            for i in range(len(b_labels)):
                im_target = np.where(im_target == b_labels[i], 100, im_target)
            print(set(b_labels))

            # 在前景上根据label的平均高度分割人体
            f_labels = list(set(un_label)-set(b_labels))
            # print(b_labels, f_labels)
            mean_h = []
            for i in range(len(f_labels)):
                pos = np.where(im_target == f_labels[i])
                mean_h.append(np.mean(pos[1]))

            part_num = cfg.CLUSTERING.PART_NUM-1
            # sorted_mean_h = np.argsort(mean_h)
            # human_part = split_n(sorted_mean_h, part_num)
            # print(human_part)
            # for i in range(part_num):
            #     for j in range(len(human_part[i])):
            #         im_target = np.where(im_target == f_labels[human_part[i][j]], 101+i, im_target)

            h_seg = (max(mean_h)-min(mean_h))/part_num
            head_labels = []
            body_labels = []
            leg_labels = []
            for i in range(len(f_labels)):
                if mean_h[i] >= min(mean_h) and mean_h[i] < (min(mean_h) + h_seg/2):
                    head_labels.append(f_labels[i])
                if mean_h[i] >= (min(mean_h) + h_seg/2) and mean_h[i] < (min(mean_h) + 2*h_seg):
                    body_labels.append(f_labels[i])
                if mean_h[i] >= (min(mean_h) + 2*h_seg):
                    leg_labels.append(f_labels[i])
            # print(head_labels, body_labels, leg_labels)
            for i in range(len(head_labels)):
                im_target = np.where(im_target == head_labels[i], 101, im_target)
            for i in range(len(body_labels)):
                im_target = np.where(im_target == body_labels[i], 102, im_target)
            for i in range(len(leg_labels)):
                im_target = np.where(im_target == leg_labels[i], 103, im_target)

            human_rgb = np.array([human_colors[(c-100) % (part_num + 1)] for c in im_target])
            human_rgb = np.concatenate([human_rgb[i] for i in range(im_target.shape[0])], axis=1)
            human_rgb = human_rgb.astype(np.uint8)

            show = np.vstack((p_image_rgb, human_rgb))
            # show = human_rgb


            if (batch_idx+1) % 2 == 0:
                frames.append(show)
                cv2.imwrite("TCL/temp/batch=%s.png" % batch_idx, show)
        if len(un_label) <= cfg.DFC.MIN_LABEL_NUM:
            break

        print('Epoch: %d, Total Loss: %f, sematic consist: %f , spatial consist: %f, bg consist: %f,Label num: %d' % (
        batch_idx, loss.item(), loss_fn(output_1d, target_1d).item(), (l_h + l_v).item(), l_img.item(), len(un_label)))

    '''save'''
    if cfg.DFC.VISULIZE:
        save_dir = 'TCL'
        # save_dir = 'cw=%s_tw=%s' % (cfg.DFC.CON_WEIGHT, cfg.DFC.TEMP_WEIGHT)
        imageio.imsave("%s/label-num=%s_id=%s_cw=%s_tw=%s.png" % (
            save_dir, cfg.DFC.MODIM2, p_id, cfg.DFC.PIXEL_WEIGHT, cfg.DFC.IMAGE_WEIGHT), show)
        imageio.mimsave("%s/label-num=%s_id=%s_cw=%s_tw=%s.gif" % (
            save_dir, cfg.DFC.MODIM2, p_id, cfg.DFC.PIXEL_WEIGHT, cfg.DFC.IMAGE_WEIGHT), frames, 'GIF', duration=0.1)

    return im_target, show


def main():
    # 从feats.pth中读取tensor数据
    p_id = 6    #4:28,  6：16， 7：54
    feats = torch.load('feats/%d.pt' % p_id)
    print(feats.shape)
    feats = feats.reshape(feats.shape[0], 64, 32, 256).transpose(0, 3, 1, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    tensor = torch.from_numpy(feats).to(device)

    # dfc
    cfg.DFC.IMAGE_WEIGHT = 1
    cfg.DFC.PIXEL_WEIGHT = 2
    cfg.DFC.DIAG_WEIGHT = 0
    cfg.DFC.MODIM2 = 32
    cfg.DFC.MIN_LABEL_NUM = 18
    cfg.DFC.EPOCH = 32
    cfg.DFC.SPATIAL = 1
    cfg.DFC.FILL = 0.5
    cfg.CLUSTERING.PART_NUM = 4

    print(tensor.shape)
    clustering(cfg, tensor, p_id)


if __name__ == '__main__':
    # cfg
    import argparse
    from config import cfg

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default='configs/HRNet32.yml', help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DFC.VISULIZE = True

    main()









