import torch


# Utility functions
def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

# Custom loss functions
def _avg_scale_ssi_L1_loss(depths_pred, depths_gt, depths_mask):
    _, height, width = depths_gt.shape
    image_size = height * width
    # transform predicted disparity to aligned depth
    target_disparity = torch.zeros_like(depths_gt)
    target_disparity[depths_mask == 1] = 1.0 / depths_gt[depths_mask == 1]

    # scale, shift = 0.00046301, 0.18316667 # midasv21-large
    # scale, shift = 0.00036764, 0.05183057 # midasv21-large-edina-train
    # scale, shift = 0.00032693, 0.66881627 # midasv21-edina-ideo-train-cameraready
    # scale, shift = 0.00027906, 0.43795583 # scannet_edina_gravity_new_edinatrainsub10_v2.pkl edina_train_sub10

    # scale, shift = 0.00020401, 0.48691764  # camready_edina_scannet
    # scale, shift = 0.03239779, 0.45822480  # camready_edina_scannet_v3_dpt_large
    # scale, shift = 0.00032525, 0.67156976  # scannet_edina_camready_midasv21_edina_train_sub10
    scale, shift = 0.05286243, 0.61732054

    # scale, shift = 0.05973897, 0.27094209 # dpt_large
    prediction_aligned = scale * depths_pred + shift
    return torch.sum(torch.abs(prediction_aligned[depths_mask == 1] - target_disparity[depths_mask == 1])) / image_size


def _ssi_L1_loss(depths_pred, depths_gt, depths_mask):
    _, height, width = depths_gt.shape
    image_size = height * width
    # transform predicted disparity to aligned depth
    target_disparity = torch.zeros_like(depths_gt)
    target_disparity[depths_mask == 1] = 1.0 / depths_gt[depths_mask == 1]

    scale, shift = compute_scale_and_shift(depths_pred, target_disparity, depths_mask)
    prediction_aligned = scale.view(-1, 1, 1) * depths_pred + shift.view(-1, 1, 1)
    return torch.sum(torch.abs(prediction_aligned[depths_mask == 1] - target_disparity[depths_mask == 1])) / image_size


def _invnoscale_L1_loss(depths_pred, depths_gt, depths_mask):
    _, height, width = depths_gt.shape
    image_size = height * width

    disparity_cap = 1.0 / 10.0
    depths_pred = depths_pred.clone()
    depths_pred[depths_pred < disparity_cap] = disparity_cap
    depths_pred = 1.0 / depths_pred

    return torch.sum(torch.abs(depths_pred[depths_mask == 1] - depths_gt[depths_mask == 1])) / image_size


def l1_loss(prediction, target, mask, reduction=reduction_batch_based):
    M = torch.sum(mask, (1, 2))
    # res = prediction - target
    # image_loss = torch.sum(mask * res * res, (1, 2))
    res = torch.abs(prediction - target)
    image_loss = torch.sum(mask * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    print('prediction:', prediction.shape)
    print('target:', target.shape)
    print('mask:', mask.shape)

    M = torch.sum(mask, (1, 2))

    print('M:', M.shape)

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    print('grad_x:', grad_x.shape)
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


def _midas_loss(prediction, target, mask):
    # Prediction is the disparity
    # disparity_cap = 1.0 / 10.0
    # prediction[prediction < disparity_cap] = disparity_cap

    print(mask)
    # Convert gt depth to gt disparity
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]

    total = l1_loss(prediction, target_disparity, mask, reduction=reduction_batch_based)

    print('l1_loss:', total)

    gradient_loss_val = 0.0
    for scale in range(4):
        step = pow(2, scale)

        gradient_loss_val += gradient_loss(prediction[:, ::step, ::step], target_disparity[:, ::step, ::step],
                                            mask[:, ::step, ::step], reduction=reduction_batch_based)

    print('gradient_loss_val:', gradient_loss_val)
    alpha = 0.5
    total += alpha * gradient_loss_val

    return total


def _L1_loss(depths_pred, depths_gt, depths_mask):
    # _, _, height, width = depths_gt.shape # baseline
    _, height, width = depths_gt.shape  # midas
    image_size = height * width
    return torch.sum(torch.abs(depths_pred[depths_mask == 1] - depths_gt[depths_mask == 1])) / image_size