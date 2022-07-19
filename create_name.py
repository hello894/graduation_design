#读取patch序列生成单幅patch来进行训练
from utils.hpatch import *
import cv2
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
from main import *

# all types of patches
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

path = "hpatches"
tskdir = "simclr\\utils"

file = open("simclr\\data\\train.txt",'w')
#Hpatches数据集的不同划分方式
with open(os.path.join(tskdir, "splits.json")) as f:
    splits = json.load(f)
splt = splits["a"]["train"]

for i in range(len(splt)):
    for j in range(len(tps)):
        im_path = os.path.join(path, splt[i], tps[j]) + '.png'
        im = cv2.imread(im_path, 0)
        N = im.shape[0] / 65
        seq = hpatch_sequence(os.path.join(path, splt[i]))
        ids = range(int(N))
        for idx in ids:
            im = get_patch(seq, tps[j], idx)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_name = splt[i] + "_" + tps[j] + "_" + str(idx)
            #避免TXT文件最后一行为空
            if i == 75 & j == 15 & idx == int(N)-1:
                file.write(("%s" % im_name))
            else:
                file.write(("%s" % im_name) + '\n')
            image.imsave("simclr\\data\\train\\" + im_name + ".png", im)
file.close()
