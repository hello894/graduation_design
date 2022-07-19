import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from extract_patches.core import extract_patches







def extract_box(kp,img):
    patch_size = 65
    size = img.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    x, y = kp.pt
    left_x = x - patch_size / 2
    right_x = x + patch_size / 2
    left_y = y - patch_size / 2
    right_y = y + patch_size / 2
    if left_x < 0:
        right_x = right_x + abs(left_x)
        left_x = 0
    if right_x > w:
        left_x = left_x - right_x + w
        right_x = w
    if left_y < 0:
        right_y = right_y + abs(left_y)
        left_y = 0
    if right_y > h:
        left_y = left_y - right_y + h
        right_y = h

    #patch = img[int(left_y):int(right_y), int(left_x):int(right_x)]
    box = [int(left_y), int(right_y), int(left_x), int(right_x)]

    return box

def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    if intersection ==0 or union == 0:
        iou = 0
    else:
        iou = intersection / union
    return iou


def test(match1, match2, kp1, kp2, img1, img2):


    box1_query = extract_box(kp1[match1.queryIdx], img1)
    box2_query = extract_box(kp1[match2.queryIdx], img1)
    iou = cal_iou_xyxy(box1_query, box2_query)
    if iou > 0.6:
        return 1
    elif iou <= 0.6:
        box1_train = extract_box(kp2[match1.trainIdx], img1)
        box2_train = extract_box(kp2[match2.trainIdx], img2)
        iou = cal_iou_xyxy(box1_train, box2_train)
        if iou > 0.6:
            return 1
        else:
            return 0

def prework(good, kp1, kp2, img1, img2):
    pre_pass = 0
    final_pass = 0
    patches1 = []
    patches2 = []
    simclr = []
    for m in range(len(good) - 2):
        for n in range(m + 1,len(good) - 1):
            pre_pass = test(good[m], good[n], kp1, kp2, img1, img2)
            if pre_pass > 0:
                final_pass += 1
        if final_pass == 0:

            patches1.append(extract_patches(kp1[good[m].queryIdx], img1, 65, 10.0, "cv2"))
            patches2.append(extract_patches(kp2[good[m].trainIdx], img2, 65, 10.0, "cv2"))
            simclr.append(good[m])
    return patches1,patches2,simclr


inlier = 0
total_point = 0

dataset = ["dataset1", "dataset2", "dataset3"]

for d in range(2,3):

    datapath = os.path.join("simclr\\data",dataset[d],"RS\\Images")
    txtpath = os.path.join("simclr\\data",dataset[d]) + ".txt"
    patchpath = os.path.join("simclr\\data\\patches",dataset[d])

    # 获取图片编号
    with open(txtpath) as f:
        filenames = f.read().split('\n')
    # png与jpg格式都有，采用遍历文件夹的方法读取
    files = os.listdir(datapath)

    for i in range(0, 2):
        print(os.path.join(patchpath, filenames[i]))
        if not os.path.isdir(os.path.join(patchpath, filenames[i])):
            os.mkdir(os.path.join(patchpath, filenames[i]))
        #读取两张待匹配整图
        img1 = cv2.imread("1.jpg")
        img2 = cv2.imread("4.jpg")

        sift = cv2.SIFT_create()
        # 获取关键点和描述符
        # kp是一个关键点列表
        # des是一个numpy数组，其大小是关键点数目乘以128,描述符，描述关键点的位置，尺度，向量等信息
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None:
            print("未检测到sift特征点")
        else:
            #定义FLANN匹配器------KD树，具体原理尚不清楚，index_params中的algorithm为0或1
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # 使用KNN算法匹配,FLANN里边就包含的KNN、KD树还有其他的最近邻算法
            matches = flann.knnMatch(des1, des2, k=2)
            # 去除错误匹配
            good = []
            for m, n in matches:  # m为第一邻近，n为第二邻近，两者距离满足以下关系才认为是正确匹配
                if m.distance <= 0.7 * n.distance:
                    good.append(m)

            patches, patches_1, simclr = prework(good, kp1, kp2, img1, img2)

            txt_index_path = os.path.join(patchpath, filenames[i]) + "index.txt"
            file = open(txt_index_path,"w")

            for j in range(0, len(patches)):
                path = os.path.join(patchpath, filenames[i], str(j)) + ".png"
                cv2.imwrite(path, patches[j])
                if j == len(patches) - 1:
                    file.write("%s" % str(kp1[simclr[j].queryIdx].pt[0]) + " " + str(kp1[simclr[j].queryIdx].pt[1]))
                else:
                    file.write("%s" % str(kp1[simclr[j].queryIdx].pt[0]) + " " + str(kp1[simclr[j].queryIdx].pt[1]) + '\n')
            txt_index_path = os.path.join(patchpath, filenames[i + 1]) + "index.txt"
            file = open(txt_index_path, "w")

            for j in range(0, len(patches_1)):
                path = os.path.join(patchpath, filenames[i + 1], str(j)) + ".png"
                cv2.imwrite(path, patches_1[j])
                if j == len(patches_1) - 1:
                    file.write("%s" % str(kp2[simclr[j].trainIdx].pt[0]) + " " + str(kp2[simclr[j].trainIdx].pt[1]))
                else:
                    file.write("%s" % str(kp2[simclr[j].trainIdx].pt[0]) + " " + str(kp2[simclr[j].trainIdx].pt[1]) + '\n')

            print("patch数目:",len(patches))
            if len(simclr) > 4:
                # 显示匹配结果
                # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
                src_pts = np.float32([kp1[m.queryIdx].pt for m in simclr]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in simclr]).reshape(-1, 1, 2)
                # findHomography 函数是计算变换矩阵
                # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
                # 返回值：M 为变换矩阵，mask是掩模
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # ravel方法将数据降维处理，最后并转换成列表格式
                matchesMask = mask.ravel().tolist()
                for m in matchesMask:
                    if m == 1:
                        inlier += 1
                total_point += len(matchesMask)
                print(inlier)
                print(total_point)
                print(inlier / total_point)
                # 获取img1的图像尺寸
                h, w, dim = img1.shape
                # pts是图像img1的四个顶点
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                if M is None:
                    print("M is None")
                else:
                    # 计算变换后的四个顶点坐标位置
                    dst = cv2.perspectiveTransform(pts, M)

                    # 根据四个顶点坐标位置在img2图像画出变换后的边框
                    # img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
                    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                       singlePointColor=None,
                                       matchesMask=matchesMask,  # draw only inliers
                                       flags=2)
                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, simclr, None, **draw_params)
                    cv2.imwrite(os.path.join(patchpath, filenames[i]) + "_sift.png", img3)


