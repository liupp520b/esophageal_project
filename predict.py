from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torchvision.utils import make_grid, save_image
from utils.color_map import decode_seg_map_sequence

from network.CPF_model.BaseNet import CPFNet
from network.CPF_model.unet import UNet
from network.dfn_models import DFN
from network.CCNet import ccnet
from network.PSPNet import pspnet
from network.Medical_Transformer import axialnet
from network.DAnet.sseg.danet import DANet
from network.axial_deeplab import axial50l
from network.TransUnet.vit_seg_modeling import VisionTransformer as Vit_seg
# from network.DAnet.sseg import danet
from network.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from network.SETR.transformer_seg import SETRModel
from network.non_local.res import Non_local

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='./datasets/data/VOCdevkit/VOC2012（食管）/test_imgs_1/imgs',
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    parser.add_argument("--other_model", type=str, default='CCNet',
                        choices=['CPF', 'DFN', 'DANet', 'CCNet', 'PSPnet', 'MedT',
                                 'axial_deeplab', 'TransUnet', 'SETR', 'non_local'], help='其他模型')
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default="./datasets/data/VOCdevkit/VOC2012（食管）/test_imgs_1/predict/our",
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0,1',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 5
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 5
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    print(opts.input)
    if os.path.isdir(opts.input):
        # for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            # print(os.path.join(opts.input, '*.png'))
            # files = glob(opts.input + r'\*', recursive=True)
        files = os.listdir(opts.input)
            # print(files)
        if len(files)>0:
            image_files.extend(files)
            # print(image_files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    
    # Set up model
    # model_map = {
    #     'deeplabv3_resnet50': network.deeplabv3_resnet50,
    #     'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    #     'deeplabv3_resnet101': network.deeplabv3_resnet101,
    #     'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    #     'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    #     'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    # }
    # model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    # if opts.separable_conv and 'plus' in opts.model:
    #     network.convert_to_separable_conv(model.classifier)
    # utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.other_model == 'CPF':
        model = CPFNet(out_planes=opts.num_classes)
    elif opts.other_model == 'UNet':
        model = UNet(in_channels=3, n_classes=opts.num_classes)
    elif opts.other_model == 'DFN':
        model = DFN.DFN(num_class=opts.num_classes)
    elif opts.other_model == 'CCNet':
        model = ccnet.ccnet(num_classes=opts.num_classes, recurrence=2)
    elif opts.other_model == 'PSPnet':
        model = pspnet.PSPNet(num_classes=opts.num_classes, backbone='resnet101', downsample_factor=16,
                              pretrained=False, aux_branch=False)
    elif opts.other_model == 'MedT':
        model = axialnet.MedT(img_size = opts.crop_size[0], imgchan = 3, num_classes = opts.num_classes)
    elif opts.other_model == 'DANet':
        model = DANet(nclass=opts.num_classes, backbone='resnet101')
    elif opts.other_model == 'axial_deeplab':
        model = axial50l(pretrained=True,num_classes = opts.num_classes)
    elif opts.other_model == 'TransUnet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = opts.num_classes
        model = Vit_seg(config_vit, img_size=512, num_classes=config_vit.n_classes)
    elif opts.other_model == 'SETR':
        model = SETRModel(patch_size=(32, 32), in_channels=3, out_channels=opts.num_classes, hidden_size=1024,
                          num_hidden_layers=24, num_attention_heads=16, decode_features=[512, 256, 128, 64])
    elif opts.other_model == 'non_local':
        model = Non_local(num_classes=opts.num_classes)


    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    if True:
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        print('start!')
        pth_path = "./checkpoints/VOC2012/block_attention(食管14)/best_deeplabv3plus_resnet101_voc_os16.pth"
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        # print("[!] Retrain")
        # print('jjjj')
        # print(model)
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose([
                T.Resize((opts.crop_size,opts.crop_size)),
                # T.RandomCrop(size=(opts.crop_size), pad_if_needed=True),
                # T.CenterCrop(opts.crop_size),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    # val_dst = VOCSegmentation(root=r"./datasets/data", year=opts.year,
    #                           image_set='test1', download=False, transform=transform)
    # val_loader = data.DataLoader(
    #     val_dst, batch_size=opts.val_batch_size, drop_last=True, shuffle=True, num_workers=0)

    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            img_path = opts.input + '/' + img_path

            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            # print(type(img))
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            # print(img_path,' ',img.size())
            # print(type(img))
            img = img.to(device)
            # images = images.to(device, dtype=torch.float32)
            # model.eval()
            pred = model(img)
            # print(pred.size())
            pred = pred[1].max(1)[1].cpu().numpy()[0] # [0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            # print(colorized_preds.size)
            colorized_preds = Image.fromarray(colorized_preds)
            # print(colorized_preds)
            # colorized_preds = make_grid(decode_seg_map_sequence(torch.max(pred[0][:3], 1)[1].detach().cpu().numpy())
            #                             , 3, normalize=False, range=(0, 255))
            # save_image(colorized_preds, os.path.join(opts.save_val_results_to, img_name+'.png'))
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name+'.png'))

if __name__ == '__main__':
    main()
