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


csvpath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\descriptors\\simclr"
datapath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\hpatches-release"
tskdir = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\simclr\\utils"
testpath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\test"

# all types of patches3
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']


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






    print('>> Please wait, loading the descriptor files...')
    # get all folders in the descriptor root folder, except the 1st which is '.'
    #path_all = [x[0] for x in os.walk(path)][1::]
    with open(os.path.join(tskdir, "splits.json")) as f:
        splits = json.load(f)
    splt = splits["a"]["test"]

    try:
        len(splt) == 40
    except:
        print(" does not seem like a valid HPatches descriptor root folder.")
    count = 0

    for p in splt:
        count = count + 1
        print(">>第%d个文件夹" %count)

        if not os.path.isdir(os.path.join(csvpath,p)):
            os.mkdir(os.path.join(csvpath,p))

        for t in tps:

            reprs = {}
            args = parser.parse_args()
            args.device = torch.device('cuda')
            model = utils.model.get_model(args)

            class TestDataset(torch.utils.data.Dataset):

                def __init__(self):
                    self.args = args
                    with open(os.path.join(os.path.join(testpath, p, t) + ".txt")) as f:
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
                        Image.open(os.path.join(testpath, p, t, self.filenames[idx]) + ".png").convert('RGB')
                    )
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
            print(">>数据装载完毕，开始提取描述符")
            simclrobj = simclr.SimCLR(model, None, dataloaders, None)
            simclrobj.load_model(args)



            reprs['test'] = simclrobj.get_representations(args, mode='test')



            #reprs['test']['X'] = (reprs['test']['X']*1000).astype(np.uint8)

            name = os.path.join(csvpath, p, t) + ".csv"


            with open(name, "w",newline='') as f:
                writer = csv.writer(f,delimiter=';')
                writer.writerows(reprs['test']['X'])


            print(name + "ok")
            #send_msg.send_m(name + "    ok")


