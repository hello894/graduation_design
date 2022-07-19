from utils.hpatch import *
import cv2
import os.path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image


# all types of patches
tps = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5',\
       't1','t2','t3','t4','t5']

path = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\hpatches-release"
tskdir = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\simclr\\utils"
testpath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\test"



with open(os.path.join(tskdir, "splits.json")) as f:
    splits = json.load(f)
splt = splits["a"]["test"]
print(len(splt))

for p in splt:
    if not os.path.isdir(os.path.join(testpath, p)):
        os.mkdir(os.path.join(testpath, p))
    for t in tps:
        if not os.path.isdir(os.path.join(testpath, p, t)):
            os.mkdir(os.path.join(testpath, p, t))
        txt_name = os.path.join(testpath, p, t) + ".txt"
        im_path = os.path.join(path, p, t) + '.png'
        file = open(txt_name,'w')
        im = cv2.imread(im_path, 0)
        N = im.shape[0] / 65
        seq = hpatch_sequence(os.path.join(path, p))
        ids = range(int(N))
        for idx in ids:
            #im = get_patch(seq, t, idx)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_name = p + "_" + t + "_" + str(idx)
            if idx != int(N)-1:
                file.write(("%s" % im_name)+'\n')
            else:
                file.write(("%s" % im_name))
            #image.imsave(os.path.join(testpath, p, t, im_name) + ".png", im)
        print(im_path)

file.close()