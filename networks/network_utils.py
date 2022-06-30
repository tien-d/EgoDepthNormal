import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from PIL import Image
import numpy as np
import cv2
import logging
import os
import sys
from datetime import datetime


def weights_init(modules, init_type='xavier'):
    assert init_type == 'xavier' or init_type == 'kaiming'
    m = modules
    if (isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or \
            isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) or
            isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear)):
        if init_type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
        for m in modules:
            weights_init(m, init_type)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ConfigureLogging(save_path):
    if save_path != '':
        filename = 'log_' + sys.argv[0] + datetime.now().strftime('_%Y%m%d_%H%M%S') + '.log'
        os.makedirs(save_path, exist_ok=True)
        full_file = os.path.join(save_path, filename)
        handlers = [logging.StreamHandler(), logging.FileHandler(full_file)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-1.1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s",
        datefmt='%m%d %H:%M:%S',
        handlers=handlers)


def SaveNormalsToImage(normals, filename):
    assert(len(normals.shape) == 3)
    assert(normals.shape[0] == 3)
    normals = normals.transpose([1, 2, 0])
    # Save in FrameNet format.
    normals = (1 + normals) * 127.5
    image = Image.fromarray(normals.astype(np.uint8))
    image.save(filename)


def SaveDepthsToImage(depths, filename):
    if len(depths.shape) == 3:
        assert(depths.shape[0] == 1)
        depths = depths.squeeze()
    assert(len(depths.shape) == 2)
    # The PNG format sometimes does not support writing to uint16 image in Pillow package, so
    # we save in uint32 format.
    image = Image.fromarray((depths * 1000).astype(np.uint32))
    image.save(filename)


def SaveMasksToImage(mask, filename):
    if len(mask.shape) == 3:
        assert(mask.shape[0] == 1)
        mask = mask.squeeze()
    assert(len(mask.shape) == 2)
    image = Image.fromarray(mask)
    image.save(filename)


def SaveRgbToImage(rgb, filename):
    if len(rgb.shape) == 4:
        assert rgb.shape[0] == 1
        rgb =rgb.squeeze()
    assert len(rgb.shape) == 3
    np_image = np.transpose(rgb * 255, axes=[1, 2, 0]).astype(np.uint8)
    image = TVF.to_pil_image(np_image, mode='RGB')
    image.save(filename)


def ComputeDepthErrorStatistics(depth_predicted, depth_gt):
    if isinstance(depth_predicted, torch.Tensor):
        depth_predicted = depth_predicted.detach().cpu().numpy()
    else:
        depth_predicted = np.array(depth_predicted)
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.detach().cpu().numpy()
    else:
        depth_gt = np.array(depth_gt)

    depth_mask = depth_gt > 0
    depth_gt = depth_gt[depth_mask]
    depth_predicted = depth_predicted[depth_mask]

    depth_ratio = np.max(depth_gt / depth_predicted, depth_predicted / depth_gt)
    depth_abs_error = (depth_gt - depth_predicted).abs()

    statistics = {
        'MAD': np.mean(depth_abs_error),
        'RMSE': np.sqrt(np.mean(depth_abs_error ** 2)),
        '1.05': 100 * np.sum(depth_ratio < 1.05) / depth_ratio.shape[0],
        '1.10': 100 * np.sum(depth_ratio < 1.10) / depth_ratio.shape[0],
        '1.25': 100 * np.sum(depth_ratio < 1.25) / depth_ratio.shape[0],
        '1.25^2': 100 * np.sum(depth_ratio < 1.25 ** 2) / depth_ratio.shape[0],
        '1.25^3': 100 * np.sum(depth_ratio < 1.25 ** 3) / depth_ratio.shape[0],
    }

    return statistics


def ComputeNormalErrorStatistics(normal_predicted, normal_gt, mask):
    assert isinstance(normal_predicted, torch.Tensor)
    assert isinstance(normal_gt, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    normal_predicted = F.normalize(normal_predicted)
    normal_gt = F.normalize(normal_gt)

    length_predicted = torch.norm(normal_predicted)
    length_gt = torch.norm(normal_gt)

    mask = mask & (length_predicted > 0.99) & (length_gt > 0.99)

    dot_products = torch.sum(normal_predicted * normal_gt, dim=1)
    dot_products = torch.clamp(dot_products, min=-1.0, max=1.0)
    angle_errors = torch.acos(dot_products) / np.pi * 180

    mask_np = mask[:, 0, :, :].detach().cpu().numpy() > 0
    angles_np = angle_errors.detach().cpu().numpy()
    normal_errors = angles_np[mask_np]


    statistics = {
        'Mean': np.average(normal_errors),
        'Median': np.median(normal_errors),
        'RMSE': np.sqrt(np.mean(normal_errors ** 2)),
        '5deg': 100 * np.sum(normal_errors < 5.0) / normal_errors.shape[0],
        '7.5deg': 100 * np.sum(normal_errors < 7.5) / normal_errors.shape[0],
        '11.25deg': 100 * np.sum(normal_errors < 11.25) / normal_errors.shape[0],
        '22.5deg': 100 * np.sum(normal_errors < 22.5) / normal_errors.shape[0],
        '30deg': 100 * np.sum(normal_errors < 30) / normal_errors.shape[0],
    }

    return statistics


def compute_depth_error(depths_gt, depths_pred, depths_mask, metrics_averaged_among_images=False):
    depth_ratio_np = torch.max(depths_gt / depths_pred, depths_pred / depths_gt).detach().cpu().numpy()
    depth_mask_np = depths_mask.detach().cpu().numpy() > 0

    if metrics_averaged_among_images:
        all_abs_rel, all_sq_rel, all_log_rmse, all_i_rmse, all_si_log = [], [], [], [], []
        all_mad, all_rmse, all_a05, all_a10, all_a1, all_a2, all_a3 = [], [], [], [], [], [], []

        depth_abs_error_no_mask = (depths_gt - depths_pred).abs().detach().cpu().numpy()
        depth_ratio_error_no_mask = depth_ratio_np
        for i in range(depth_abs_error_no_mask.shape[0]):
            if depth_mask_np[i].sum() > 0:
                depth_abs_error_image_i = depth_abs_error_no_mask[i][depth_mask_np[i]]
                depth_ratio_error_image_i = depth_ratio_error_no_mask[i][depth_mask_np[i]]

                # additional metrics
                pr = depths_pred[i, ...].squeeze().detach().cpu().numpy()[depth_mask_np[i].squeeze()]
                gt = depths_gt[i, ...].squeeze().detach().cpu().numpy()[depth_mask_np[i].squeeze()]

                abs_rel = np.mean(np.abs(gt - pr) / gt)
                sq_rel = np.mean(((gt - pr) ** 2) / gt)
                rmse_log = (np.log(gt) - np.log(pr)) ** 2
                rmse_log = np.sqrt(rmse_log.mean())

                i_rmse = (1 / gt - 1 / (pr + 1e-4)) ** 2
                # i_rmse = (1 / gt - 1 / pr) ** 2
                i_rmse = np.sqrt(i_rmse.mean())

                # sc_inv
                log_diff = np.log(gt) - np.log(pr)
                num_pixels = np.float32(log_diff.size)
                sc_inv = np.sqrt(
                    np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(
                        num_pixels))

                all_abs_rel.append(abs_rel)
                all_sq_rel.append(sq_rel)
                all_log_rmse.append(rmse_log)
                all_i_rmse.append(i_rmse)
                all_si_log.append(sc_inv)

                all_mad.append(np.mean(depth_abs_error_image_i).item())
                all_rmse.append(np.sqrt(np.mean(depth_abs_error_image_i ** 2)).item())

                all_a05.append(
                    100. * np.sum(depth_ratio_error_image_i < 1.05).item() / depth_ratio_error_image_i.shape[0])
                all_a10.append(
                    100. * np.sum(depth_ratio_error_image_i < 1.10).item() / depth_ratio_error_image_i.shape[0])
                all_a1.append(
                    100. * np.sum(depth_ratio_error_image_i < 1.25).item() / depth_ratio_error_image_i.shape[0])
                all_a2.append(100. * np.sum(depth_ratio_error_image_i < 1.25 ** 2).item() /
                              depth_ratio_error_image_i.shape[0])
                all_a3.append(100. * np.sum(depth_ratio_error_image_i < 1.25 ** 3).item() /
                              depth_ratio_error_image_i.shape[0])

        depth_error_every_image = {'abs_rel': np.array(all_abs_rel),
                                   'sq_rel': np.array(all_sq_rel),
                                   'rmse_log': np.array(all_log_rmse),
                                   'i_rmse': np.array(all_i_rmse),
                                   'sc_inv': np.array(all_si_log),
                                   'mad': np.array(all_mad),
                                   'rmse': np.array(all_rmse),
                                   '1.05': np.array(all_a05),
                                   '1.1': np.array(all_a10),
                                   '1.25': np.array(all_a1),
                                   '1.25^2': np.array(all_a2),
                                   '1.25^3': np.array(all_a3)}
        return depth_error_every_image
    else:
        depth_ratio_error = depth_ratio_np[depth_mask_np]
        depth_abs_error = (depths_gt - depths_pred).abs().detach().cpu().numpy()[depth_mask_np]
        return depth_ratio_error, depth_abs_error


def GetDepthPrintableRatios(valid_gt, valid_preds):
    ratios = torch.max(valid_preds / valid_gt, valid_gt / valid_preds)
    ratios[ratios <= 0] = float('inf')

    r05 = 100 * torch.sum(ratios < 1.05).item() / (ratios.nelement() + 1e-7)
    r10 = 100 * torch.sum(ratios < 1.10).item() / (ratios.nelement() + 1e-7)
    r25 = 100 * torch.sum(ratios < 1.25).item() / (ratios.nelement() + 1e-7)
    r25_2 = 100 * torch.sum(ratios < 1.25**2).item() / (ratios.nelement() + 1e-7)
    r25_3 = 100 * torch.sum(ratios < 1.25**3).item() / (ratios.nelement() + 1e-7)

    return {'D_1.05': round(r05, 1), 'D_1.10': round(r10, 1), 'D_1.25': round(r25, 1),
            'D_1.56': round(r25_2, 1), 'D_1.95': round(r25_3, 1)}


def create_color_error_depth_image(depth_error, mask, thres=2.0):
    valid_mask_depth = mask > 0
    threshold_mask_5m = depth_error >= thres
    depth_error[threshold_mask_5m] = thres

    output_color_depth_img = cv2.applyColorMap(np.uint8(depth_error * 255 / thres), cv2.COLORMAP_JET)
    output_color_depth_img = cv2.cvtColor(output_color_depth_img, cv2.COLOR_RGB2BGR)
    output_color_depth_img[:, :, 0][~valid_mask_depth] = 128
    output_color_depth_img[:, :, 1][~valid_mask_depth] = 128
    output_color_depth_img[:, :, 2][~valid_mask_depth] = 128
    return output_color_depth_img


def create_color_depth_image(depth, thres=5.0):
    valid_mask_depth = depth > 0
    threshold_mask_5m = depth >= thres
    depth[threshold_mask_5m] = thres

    # output_color_depth_img = cv2.applyColorMap(np.uint8(depth * 255 / thres), cv2.COLORMAP_JET)
    output_color_depth_img = cv2.applyColorMap(255 - np.uint8(depth * 255 / thres), cv2.COLORMAP_MAGMA)
    output_color_depth_img = cv2.cvtColor(output_color_depth_img, cv2.COLOR_RGB2BGR)
    # output_color_depth_img[:, :, 0][~valid_mask_depth] = 0
    # output_color_depth_img[:, :, 1][~valid_mask_depth] = 0
    # output_color_depth_img[:, :, 2][~valid_mask_depth] = 139
    output_color_depth_img[:, :, 0][~valid_mask_depth] = 128
    output_color_depth_img[:, :, 1][~valid_mask_depth] = 128
    output_color_depth_img[:, :, 2][~valid_mask_depth] = 128
    return output_color_depth_img


def create_color_error_normal_image(normal_error, mask, angle_thres=20.):
    valid_mask_img = mask > 0

    normal_error[~valid_mask_img] = angle_thres
    normal_error[normal_error > angle_thres] = angle_thres

    output_error_img = cv2.applyColorMap(np.uint8(normal_error * 255 / angle_thres), cv2.COLORMAP_JET)
    output_error_img = cv2.cvtColor(output_error_img, cv2.COLOR_RGB2BGR)
    output_error_img[:, :, 0][~valid_mask_img] = 128
    output_error_img[:, :, 1][~valid_mask_img] = 128
    output_error_img[:, :, 2][~valid_mask_img] = 128

    return output_error_img


def create_color_normal_image(normal, mask=None, color='gray'):
    output_normal_img = (1.0 + normal) * 127.5
    output_normal_img = output_normal_img.astype(np.uint8)
    if mask is not None:
        valid_mask_img = mask > 0
        if color == 'gray':
            output_normal_img[:, :, 0][~valid_mask_img] = 128
            output_normal_img[:, :, 1][~valid_mask_img] = 128
            output_normal_img[:, :, 2][~valid_mask_img] = 128
        elif color == 'black':
            output_normal_img[:, :, 0][~valid_mask_img] = 0
            output_normal_img[:, :, 1][~valid_mask_img] = 0
            output_normal_img[:, :, 2][~valid_mask_img] = 0
    return output_normal_img

