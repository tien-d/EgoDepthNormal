import torch
import torch.nn
import torch.nn.functional as F
import numpy as np


def compute_normal_vectors_loss_l2(norm_gt, pred_normals, mask):
    norm1 = pred_normals[:, 0:3, :, :]

    loss = -torch.sum(F.cosine_similarity(norm1, norm_gt, dim=1)) / torch.sum(mask)

    norm1 = F.normalize(norm1)
    angle = torch.acos(torch.clamp(torch.sum(norm1 * norm_gt, dim=1), -1, 1)) / np.pi * 180
    angle = angle.view(mask.shape[0], 1, mask.shape[2], mask.shape[3]) * mask
    angle = torch.sum(angle)

    return loss, angle


def compute_robust_acos_loss(norm_gt, pred_normals, mask):
    mask = mask > 0
    prediction_error = torch.cosine_similarity(pred_normals, norm_gt, dim=1, eps=1e-6)

    # Robust acos loss
    acos_mask = mask.float() \
                * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
    cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
    acos_mask = acos_mask > 0.0
    cos_mask = cos_mask > 0.0
    loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])

    return loss


def compute_normal_vectors_loss_l1(norm_gt, pred_normals, mask, normalize_prediction=True):
    mask = mask.float()
    loss_func = torch.nn.L1Loss(reduction='sum')
    if normalize_prediction:
        norms = Normalize(pred_normals[:, 0:3, :, :])
    else:
        norms = pred_normals[:, 0:3, :, :]

    angle = torch.acos(torch.clamp(torch.sum(norms * norm_gt, dim=1), -1, 1)) / np.pi * 180
    angle = angle.view(mask.shape[0], 1, mask.shape[2], mask.shape[3]) * mask
    angle = torch.sum(angle)

    num_elements = torch.sum(mask).item()
    loss = loss_func(norms * mask, norm_gt * mask) / num_elements
    return loss, angle


# Utility functions
def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def l1_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    # res = prediction - target
    # image_loss = torch.sum(mask * res * res, (1, 2))
    res = torch.abs(prediction - target)
    # print('res:', res.shape)
    image_loss = torch.sum(mask.unsqueeze(1).repeat(1, 3, 1, 1) * res, (1, 2, 3))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    # print('prediction:', prediction.shape)
    # print('target:', target.shape)
    # print('mask:', mask.shape)

    M = torch.sum(mask, (1, 2))

    # print('M:', M.shape)

    mask_dup = mask.unsqueeze(1).repeat(1, 3, 1, 1)
    # calculate norm
    prediction_norm = torch.linalg.norm(prediction, ord=2, dim=1)
    prediction_norm = prediction_norm.unsqueeze(1).repeat(1, 3, 1, 1)
    prediction_norm = prediction_norm.detach()
    # print('prediction_norm:', prediction_norm)

    diff = prediction - prediction_norm * target
    # print('diff:', diff.shape)
    diff = torch.mul(mask_dup, diff)

    grad_x = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1])
    mask_x = torch.mul(mask_dup[:, :, :, 1:], mask_dup[:, :, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)
    # print('grad_x:', grad_x.shape)

    grad_y = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
    mask_y = torch.mul(mask_dup[:, :, 1:, :], mask_dup[:, :, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)
    # print('grad_y:', grad_y.shape)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    return reduction(image_loss, M)


def _midas_loss(prediction, target, mask):
    # Prediction is the disparity
    # disparity_cap = 1.0 / 10.0
    # prediction[prediction < disparity_cap] = disparity_cap

    # # Convert gt depth to gt disparity
    # target_disparity = torch.zeros_like(target)
    # target_disparity[mask == 1] = 1.0 / target[mask == 1]

    total = l1_loss(prediction, target, mask, reduction=reduction_batch_based)

    # print('l1_loss:', total)

    gradient_loss_val = 0.0
    for scale in range(4):
        step = pow(2, scale)

        gradient_loss_val += gradient_loss(prediction[:, :, ::step, ::step], target[:, :, ::step, ::step],
                                            mask[:, ::step, ::step], reduction=reduction_batch_based)

    # print('gradient_loss_val:', gradient_loss_val)
    alpha = 0.5
    total += alpha * gradient_loss_val

    return total
