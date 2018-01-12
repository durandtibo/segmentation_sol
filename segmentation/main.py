import argparse
import os
import sys
from os.path import exists

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from tqdm import tqdm

from segmentation.model import resnet_50_hc5_psp

parser = argparse.ArgumentParser(description='Segmentation')
parser.add_argument('--image', type=str, metavar='DIR', help='path to the image directory or image filename')
parser.add_argument('--checkpoint_dir', type=str, metavar='FILE', help='checkpoint directory', default='model')
parser.add_argument('--max_size', default=1000, type=int, metavar='SIZE', help='smallest image size')
parser.add_argument('--output_dir', type=str, metavar='DIR', help='directory where to save the results',
                    default='outputs')

checkpoint_name = 'checkpoint_resnet_50_hc5_psp.pth.tar'


def load_checkpoint(model, checkpoint_dir):
    checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_name)
    if not exists(checkpoint_filename):
        print('{filename} does not exist'.format(filename=checkpoint_filename))
        sys.exit(-1)

    print("=> loading checkpoint '{}'".format(checkpoint_filename))
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_list_images(filename):
    filenames = list()
    if os.path.isdir(filename):
        files = sorted(os.listdir(filename))
        for file in files:
            filenames.append(os.path.join(filename, file))
    else:
        filenames.append(filename)
    return filenames


def predict_mask(model, filename, output_dir, transform=None, verbose=False):
    extension = '.png'
    file, file_extension = os.path.splitext(filename)
    mask_name = os.path.basename(file + extension)
    mask_filename = os.path.join(output_dir, mask_name)

    if not os.path.exists(mask_filename):

        # load image
        image = Image.open(filename).convert('RGB')
        if transform is not None:
            image = transform(image)
        image = torch.unsqueeze(image, 0)

        # conversion to variable
        input_var = torch.autograd.Variable(image, volatile=True)

        # compute output
        output = model(input_var)

        # predict mask
        mask = torch.max(output, 1)[1][0]

        if mask.is_cuda:
            mask = mask.cpu()
        mask = np.asarray(mask.data.numpy(), dtype='uint8')
        mask = Image.fromarray(mask, mode='P')
        mask.putpalette([
            0, 255, 0,
            218, 165, 32,
            139, 69, 19,
            255, 255, 255,
        ])

        parent_dir = os.path.dirname(mask_filename)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        if verbose:
            print('saving predicted mask in {filename}'.format(filename=mask_filename))
        mask.save(mask_filename)


def main():
    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir

    # define model
    model = resnet_50_hc5_psp(num_classes=4, pretrained='imagenet', dim=256, dropout_p=0.1, fusion='sum')

    # load checkpoint
    model = load_checkpoint(model, checkpoint_dir)
    model.eval()

    # define preprocessing
    normalize = transforms.Normalize(mean=model.image_normalization_mean, std=model.image_normalization_std)
    transform = transforms.Compose([
        Resize(args.max_size),
        transforms.ToTensor(),
        normalize,
    ])

    image_filenames = get_list_images(args.image)
    print('Find {nb} images to process'.format(nb=len(image_filenames)))

    image_filenames = tqdm(image_filenames, desc='Prediction')

    for filename in image_filenames:
        predict_mask(model=model, filename=filename, output_dir=output_dir, transform=transform)


if __name__ == '__main__':
    global args, use_gpu
    args = parser.parse_args()
    main()
