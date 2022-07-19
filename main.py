import argparse
import torch
import torchvision
import utils
import simclr
from PIL import Image
import os

# making a command line interface

#为了方便运行改变了地址变量
# datapath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data"
# respath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\simclr\\results"
# tskdir = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\simclr\\utils"
# all types of patches
tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5', 't1', 't2', 't3', 't4', 't5']

#也可以在命令行写入参数
parser = argparse.ArgumentParser(
    description="This is the command line interface for the SimCLR framework for self-supervised learning. Below are the arguments which are required to run this program.")

parser.add_argument('-bs', '--batch_size', default=128, type=int, help="The batch size for self-supervised training")
parser.add_argument('-nw', '--num_workers', default=0, type=int, help="The number of workers for loading data")
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('--multiple_gpus', action='store_true')


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        with open(os.path.join(datapath, "train.txt")) as f:
            self.filenames = f.read().split('\n')

    def __len__(self):
        return len(self.filenames)

    def tensorify(self, img):
        return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
            torchvision.transforms.ToTensor()(img)
        )

    def augmented_image(self, img):
        return utils.transforms.get_color_distortion(1)(
            torchvision.transforms.RandomResizedCrop(65)(img)
        )

    def __getitem__(self, idx):
        img = torchvision.transforms.Resize((65, 65))(
            Image.open(os.path.join(datapath, 'train', self.filenames[idx]) + ".png").convert('RGB')
        )

        return {
            'image1': self.tensorify(
                self.augmented_image(img)
            ),
            'image2': self.tensorify(
                self.augmented_image(img)
            )
        }


if __name__ == '__main__':

    args = parser.parse_args()
    args.device = torch.device('cuda')

    model = utils.model.get_model(args)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(
        TrainDataset(args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    loss_fn = utils.ntxent.loss_function

    simclrobj = simclr.SimCLR(model, optimizer, dataloaders, loss_fn)
    simclrobj.train(args, 20, 10)
