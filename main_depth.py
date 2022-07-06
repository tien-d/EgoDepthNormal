import argparse
import logging

import cv2
import numpy as np
import os
import sys
import torch
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import network_run
import pickle
import networks.network_utils as network_utils
from PIL import Image
from dataloaders.custom_dataloader import CustomDataset
from dataloaders.scannet_edina_dataloader import ScanNetEdinaMultiRectsCropNoResizeDataset
from networks.midas.transforms import Resize, NormalizeImage, PrepareForNet
from networks.midas.midas_net_depth import MidasNetDepth
from networks.midas.dpt_depth import DPTDepthModel
from networks.fpn.depth_fpn import ExtendedFPNDepth
from networks.fpn.depth_dfpn import DFPNDepth
import re
import depth_utils


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--data_root', type=str,
                        help='Location of the dataset.')
    parser.add_argument('--checkpoint', type=str,
                        help='Location of the checkpoint to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--train_usage', type=str, default='train',
                        help='framenet_train/edina_train/train - all.')
    parser.add_argument('--test_usage', type=str, default='test',
                        help='framenet_train/edina_train/train - all.')

    parser.add_argument('--train_mode', type=str, default='rectified',
                        help='Train modes are standard/rectified/augmentation.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--save_visualization', type=str, default='',
                        help='Saving network output images.')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--epoch', type=int, default=0,
                        help='The epoch to resume training from.')
    parser.add_argument('--iter', type=int, default=0,
                        help='The iteration to resume training from.')

    parser.add_argument('--dataset_type', type=str, default='scannet')
    parser.add_argument('--dataset_pickle_file', type=str,
                        default='/mars/mnt/oitstorage/tien_storage/tien_workspace_storage/SpatialRectifiers/pickle_files/scannet_edina_gravity.pkl')

    parser.add_argument('--input_dir', type=str,
                        default='/mars/mnt/oitstorage/tien_storage/tien_workspace_storage/SpatialRectifiers/pickle_files/scannet_edina_gravity.pkl')

    parser.add_argument('--cluster_index', type=int, default=0)

    parser.add_argument('--dataloader_test_workers', type=int, default=4)
    parser.add_argument('--dataloader_train_workers', type=int, default=4)

    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save_every_n_iteration', type=int, default=2000,
                        help='Save a checkpoint every n iterations (iterations reset on new epoch).')
    parser.add_argument('--save_every_n_epoch', type=int, default=1,
                        help='Save a checkpoint on the first iteration of every n epochs (independent of iteration).')
    parser.add_argument('--eval_test_every_n_iterations', type=int, default=2000,
                        help='Evaluate the network on the test set every n iterations when in training.')

    parser.add_argument('--enable_multi_gpu', type=int, default=0,
                        help='If nonzero, use all available GPUs.')

    parser.add_argument('--skip_every_n_image_test', type=int, default=20,
                        help='Skip every n image in the test split.')
    parser.add_argument('--skip_every_n_image_train', type=int, default=1,
                        help='Skip every n image in the test split.')

    parser.add_argument('--resnet_arch', type=int, default=18,
                        help='ResNet architecture for ModifiedFPN (18/34/50/101/152)')
    parser.add_argument('--use_spatial_rectifier', type=int, default=0,
                        help='Allow network to use spatial rectifier')

    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Maximum number of epochs for training.')

    parser.add_argument('--model_type', type=str, default='dpt_large')

    parser.add_argument('--metrics_averaged_among_images', type=int, default=1,
                        help='Which type of metric we are computing.')

    return parser.parse_args()


class RunDepthEstimation(network_run.DefaultImageNetwork):
    def __init__(self, arguments, train_dataloader, test_dataloader, network_class_creator):
        super(RunDepthEstimation, self).__init__(arguments, train_dataloader, test_dataloader,
                                                      network_class_creator=network_class_creator,
                                                      estimates_depth=True, estimates_normal=False)
        # Make a local copy of configuration file
        self.args = arguments

        self.output_path = arguments.save_visualization
        if self.output_path != '':
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)
        self.save_idx = 0

        # Train mode
        self.train_mode = arguments.train_mode

        # Training or testing
        self.is_train = arguments.train

    def visualization(self, input_batch, rectified_batch, output_pred):
        output_path = self.output_path
        for i in range(input_batch['image'].shape[0]):
            out_img_l1 = []

            # start viz !!!
            rgb = input_batch['image-original'][i].squeeze().detach().cpu().numpy()
            rgb_image = (rgb * 255).astype(np.uint8)
            out_img_l1.append(rgb_image)

            depth_pred_img = network_utils.create_color_depth_image(output_pred[i, ...].squeeze().detach().cpu().numpy(), thres=3.0)
            out_img_l1.append(depth_pred_img)

            if self.is_train:
                depth_gt_img = network_utils.create_color_depth_image(input_batch['depth'][i, ...].squeeze().detach().cpu().numpy(), thres=3.0)
                out_img_l1.append(depth_gt_img)

                depth_error_img = network_utils.create_color_error_depth_image(
                    (output_pred[i, ...] - input_batch['depth'][i, ...]).abs().squeeze().detach().cpu().numpy(),
                    (input_batch['normal-mask'][i, ...] > 0).squeeze().detach().cpu().numpy(), thres=1.0)
                out_img_l1.append(depth_error_img)

            out_img = np.concatenate(out_img_l1, axis=1)

            image = Image.fromarray(out_img.astype(np.uint8))
            original_filename, original_ext = os.path.splitext(os.path.basename(input_batch['image-name'][i]))
            image.save(os.path.join(output_path, 'viz_{0}_{1:06d}.png'.format(original_filename, self.save_idx)))

            self.save_idx += 1

    def _prepare_data(self, input_batch):
        # input_batch = {data_key: input_batch[data_key].cuda(non_blocking=True) for data_key in input_batch}
        for data_key in input_batch:
            if data_key != 'image-name':
                input_batch[data_key] = input_batch[data_key].cuda(non_blocking=True)
            else:
                input_batch[data_key] = input_batch[data_key]

        if self.args.train_mode == 'rectified' and 'a_q_g' in input_batch:
            q = input_batch['a_q_g']
            rectified_batch = self.warping_module.warp_all_with_quaternion_center_aligned(input_batch, q)

            # Remove those that have tiny mask from computing the loss function
            # image_area = rectified_batch['normal-mask'].shape[1] * rectified_batch['normal-mask'].shape[2]
            # mask_areas = torch.sum((rectified_batch['normal-mask'] > 0).float(), dim=(1, 2)) / image_area
            rectified_batch['normal-mask'][(torch.abs(q[:, 0]) > 0.2) |
                                           (torch.abs(q[:, 1]) > 0.2)] = 0.0
            return rectified_batch
        else:
            return input_batch

    def _call_cnn(self, input_batch):
        # input_batch = {data_key: input_batch[data_key].cuda(non_blocking=True) for data_key in input_batch}
        input_batch = self._prepare_data(input_batch)

        rgb_image = input_batch['image']
        depths_pred = self.cnn.forward(rgb_image)

        if depths_pred.shape[1] == 1:
            depths_pred = depths_pred.squeeze(1)

        # resize prediction to match target
        if 'normal-mask' in input_batch and depths_pred.shape[-2:] != input_batch["normal-mask"].shape[-2:]:
            logging.warning('Prediction will be resized to match original mask! Please double check and disable this warning if this is intended!')
            depths_pred = F.interpolate(
                depths_pred.unsqueeze(1),
                size=input_batch["normal-mask"].shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            depths_pred = depths_pred.squeeze(1)  # (bs, h, w)

        cnn_outputs = {'d': depths_pred, 'rectified_input': input_batch}

        if self.output_path != '':
            depths_pred = self._get_network_output_depth(cnn_outputs, output_type='invdepth_noscale')  # invnoscale
            self.visualization(input_batch, None, depths_pred)

        return cnn_outputs

    def _get_network_output_depth(self, network_output, output_type):
        assert(self._network_estimates_depth())
        prediction = network_output['d']
        if output_type == 'invdepth_noscale':
            disparity_cap = 1.0 / 10.0
            depths_pred = prediction.clone()
            depths_pred[depths_pred < disparity_cap] = disparity_cap
            depths_pred = 1.0 / depths_pred
        elif output_type == 'invdepth_gtscale':
            # scale with gt_depth
            input_batch = self._get_network_augmented_input(network_output)
            target = input_batch['depth']
            mask = input_batch['normal-mask']

            # transform gt depth to inverse depth
            target_disparity = torch.zeros_like(target)
            target_disparity[mask == 1] = 1.0 / target[mask == 1]

            scale, shift = depth_utils.compute_scale_and_shift(prediction, target_disparity, mask)
            prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

            disparity_cap = 1.0 / 10.0
            prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

            depths_pred = 1.0 / prediction_aligned
        elif output_type == 'direct':
            depths_pred = prediction
        else:
            raise Exception('Output type not supported!')

        return depths_pred

    def _get_network_augmented_input(self, network_output):
        return network_output['rectified_input']

    def _network_loss(self, input_batch, cnn_outputs):
        losses_map = {}
        other_outputs = {}

        if self._network_estimates_depth():
            depths_gt = self._get_network_augmented_input(cnn_outputs)['depth']
            depth_mask = self._get_network_augmented_input(cnn_outputs)['normal-mask']
            depths_pred = self._get_network_output_depth(cnn_outputs, output_type='direct')

            # losses_map['L1'] = _L1_loss(depths_pred, depths_gt, depth_mask)
            # losses_map['ssi'] = depth_utils._ssi_L1_loss(depths_pred, depths_gt, depth_mask)
            # losses_map['invnoscale'] = _invnoscale_L1_loss(depths_pred, depths_gt, depth_mask)
            losses_map['midas_loss'] = depth_utils._midas_loss(depths_pred, depths_gt, depth_mask)
            # losses_map['avgscale_ssi'] = _avg_scale_ssi_L1_loss(depths_pred, depths_gt, depth_mask)
        return losses_map, other_outputs

    def _network_evaluate(self, input_batch, cnn_outputs):
        normal_error = None
        depth_ratio_error = None
        depth_abs_error = None

        if self._network_estimates_depth():
            depths_gt = self._get_network_augmented_input(cnn_outputs)['depth']
            depth_mask = self._get_network_augmented_input(cnn_outputs)['normal-mask']
            depths_pred = self._get_network_output_depth(cnn_outputs, output_type='invdepth_noscale')

            if self.args.metrics_averaged_among_images:
                depth_error_every_image = network_utils.compute_depth_error(depths_gt=depths_gt,
                                                                            depths_pred=depths_pred,
                                                                            depths_mask=depth_mask,
                                                                            metrics_averaged_among_images=True)
                return depth_error_every_image
            else:
                depth_ratio_error, depth_abs_error = network_utils.compute_depth_error(depths_gt=depths_gt,
                                                                                       depths_pred=depths_pred,
                                                                                       depths_mask=depth_mask,
                                                                                       metrics_averaged_among_images=False)

        return normal_error, depth_ratio_error, depth_abs_error


def prepare_network(model_type, input_size):
    _SUPPORTED_MODELS = ['dpt_large', 'midas_v21', 'efpn', 'dfpn']
    assert model_type in _SUPPORTED_MODELS, 'Model is not supported!'

    net_w, net_h = input_size[0], input_size[1]

    if model_type == "dpt_large":  # DPT-Large
        net_creator = lambda: DPTDepthModel(
            path=None,
            backbone="vitl16_384",
            non_negative=True,
        )
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21":
        net_creator = lambda: MidasNetDepth(
            None,
            non_negative=True)
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif args.model_type == 'efpn':
        net_creator = lambda: ExtendedFPNDepth(args)
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif args.model_type == 'dfpn':
        net_creator = lambda: DFPNDepth()
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return net_creator, transform


if __name__ == '__main__':
    args = ParseCmdLineArguments()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    network_utils.ConfigureLogging(args.save)

    # First log all the arguments and the values for the record.
    logging.info('sys.argv = {}'.format(sys.argv))
    logging.info('parsed arguments and their values: {}'.format(vars(args)))

    # load network and transform
    input_w, input_h = 640, 480
    net_creator, transform = prepare_network(model_type=args.model_type,
                                             input_size=(input_w, input_h))
    logging.info(f'Input image will be resized to {input_w}x{input_h}')

    if args.dataset_type == 'demo':
        train_dataloader = None

        test_dataset = CustomDataset(image_dir=args.input_dir,
                                     glob_patterns=['*.png', '*.jpg', '*.jpeg'],
                                     skip_every_n_image=args.skip_every_n_image_test,
                                     transform=transform,
                                     size=(input_h, input_w))

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_test_workers,
                                     pin_memory=True)
    else:
        train_dataset = ScanNetEdinaMultiRectsCropNoResizeDataset(data_root=args.data_root,
                                                                  usage=args.train_usage,
                                                                  skip_every_n_image=args.skip_every_n_image_train,
                                                                  dataset_pickle_file=args.dataset_pickle_file,
                                                                  transform=transform,
                                                                  size=(input_h, input_w))

        test_dataset = ScanNetEdinaMultiRectsCropNoResizeDataset(data_root=args.data_root,
                                                                 usage=args.test_usage,
                                                                 dataset_pickle_file=args.dataset_pickle_file,
                                                                 skip_every_n_image=args.skip_every_n_image_test,
                                                                 transform=transform,
                                                                 size=(input_h, input_w))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.dataloader_train_workers,
                                      pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.dataloader_test_workers,
                                     pin_memory=True)

    network = RunDepthEstimation(args, train_dataloader, test_dataloader, network_class_creator=net_creator)

    # Main process
    # Check if this is training or testing.
    if args.train != 0:
        logging.info('Training the network.')
        if args.epoch != 0:
            resume_model = os.path.join(args.save,
                                        'model-epoch-{0:05d}-iter-{1:05d}.ckpt'.format(args.epoch, args.iter))
            network.load_network_from_file(resume_model)
        elif args.checkpoint:
            network.load_network_from_file(args.checkpoint)
        if args.save == '':
            logging.warning('NO CHECKPOINTS WILL BE SAVED! SET --save FLAG TO SAVE TO A DIRECTORY.')
        network.train(starting_epoch=args.epoch, max_epochs=args.max_epochs)
    else:
        assert args.checkpoint is not None
        network.load_network_from_file(args.checkpoint)
        network.evaluate(metrics_averaged_among_images=args.metrics_averaged_among_images)
