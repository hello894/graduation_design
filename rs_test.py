import argparse
import torch
import os
import torchvision
import utils
import simclr
from PIL import Image
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import csv
from utils.hpatch import *
import send_msg
import time


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        print(kp1[img1_idx])
        (x1,y1) = kp1[img1_idx][0][0],kp1[img1_idx][0][1]
        (x2,y2) = kp2[img2_idx][0][0],kp2[img2_idx][0][1]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1

        print((x1,y1))
        cv2.circle(out, ((int(x1)),int(y1)), 2, (0, 0, 255), 1)      #画圆，cv2.circle()参考官方文档
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (0,0,255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0,0,255), 1, lineType=cv2.LINE_AA, shift=0)  #画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out


# making a command line interface
parser = argparse.ArgumentParser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('-datapath', type=str,help="Path to the data root folder which contains train and test folders")

parser.add_argument('-model_path', default="D:\\graduation design\\SIMCLR-pytorch\\model.pth",type=str, help="Path to the trained self-supervised model")

parser.add_argument('-respath',default="D:\graduation design\SIMCLR-pytorch\data\milli_imagenet", type=str, help="Path to the results where the evaluation metrics would be stored. ")

parser.add_argument('-bs','--batch_size',default=128, type=int, help="The batch size for evaluation")

parser.add_argument('-nw','--num_workers',default=0,type=int,help="The number of workers for loading data")

parser.add_argument('-c','--cuda',action='store_true')

parser.add_argument('--multiple_gpus', action='store_true')

parser.add_argument('--remove_top_layers', default=1, type=int)



if __name__ == '__main__':
    files1 = os.listdir("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000019")

    files1.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序


    print(len(files1))
    reprs = {}
    args = parser.parse_args()
    args.device = torch.device('cuda')
    model = utils.model.get_model(args)


    class TestDataset(torch.utils.data.Dataset):

        def __init__(self):
            self.args = args
            self.files = files1

        def __len__(self):
            return len(self.files)

        def tensorify(self, img):
            return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                torchvision.transforms.ToTensor()(img)
            )

        def __getitem__(self, idx):
            img = torchvision.transforms.Resize((65, 65))(Image.open(
                os.path.join(
                    "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000019", self.files[idx])).convert('RGB'))

            return {
                'image': self.tensorify(img),
            }


    dataloaders = {}

    dataloaders['test'] = torch.utils.data.DataLoader(
        TestDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    simclrobj = simclr.SimCLR(model, None, dataloaders, None)
    simclrobj.load_model(args)

    reprs['test'] = simclrobj.get_representations(args, mode='test')
    print(reprs['test']['X'].shape)
    print("len(reprs['test']['X']):", len(reprs['test']['X']))
    with open("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\descriptors\\RS\\dataset3\\00000019.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(reprs['test']['X'].astype(np.float32))

    files2 = os.listdir("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000020")

    files2.sort(key=lambda x: int(x[:-4]))  # 将'.jpg'左边的字符转换成整数型进行排序



    reprs = {}
    args = parser.parse_args()
    args.device = torch.device('cuda')
    model = utils.model.get_model(args)


    class TestDataset(torch.utils.data.Dataset):

        def __init__(self):
            self.args = args
            self.files = files2

        def __len__(self):
            return len(self.files)

        def tensorify(self, img):
            return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(
                torchvision.transforms.ToTensor()(img)
            )

        def __getitem__(self, idx):
            img = torchvision.transforms.Resize((65, 65))(Image.open(
                os.path.join(
                    "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000020", self.files[idx])).convert('RGB'))

            return {
                'image': self.tensorify(img),
            }


    dataloaders = {}

    dataloaders['test'] = torch.utils.data.DataLoader(
        TestDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    simclrobj = simclr.SimCLR(model, None, dataloaders, None)
    simclrobj.load_model(args)

    reprs['test'] = simclrobj.get_representations(args, mode='test')
    print(reprs['test']['X'].shape)
    print("len(reprs['test']['X']):", len(reprs['test']['X']))
    with open("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\descriptors\\RS\\dataset3\\00000020.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerows(reprs['test']['X'].astype(np.float32))



















