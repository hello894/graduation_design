import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt
from extract_patches.core import extract_patches
from PIL import Image


dataset = ["dataset1", "dataset2", "dataset3"]

for d in range(2, 3):

    datapath = os.path.join("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data", dataset[d], "RS\\Images")
    txtpath = os.path.join("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data", dataset[d]) + ".txt"
    patchpath = os.path.join("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches", dataset[d])

    # 获取图片编号
    with open(txtpath) as f:
        filenames = f.read().split('\n')
    # png与jpg格式都有，采用遍历文件夹的方法读取
    files = os.listdir(datapath)

    for i in range(1, 2):
        print(os.path.join(patchpath, filenames[i]))
        if not os.path.isdir(os.path.join(patchpath, filenames[i])):
            os.mkdir(os.path.join(patchpath, filenames[i]))


        img1 = cv2.imread("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\dataset1\\RS\\Images\\00000001.png")
        img2 = cv2.imread("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\dataset1\\RS\\Images\\00000002.png")

        # img1 = cv2.imread("D:\\download\\hpatches-sequences-release\\v_artisans\\3.ppm")
        # img2 = cv2.imread("D:\\download\\hpatches-sequences-release\\v_artisans\\4.ppm")


        sift = cv2.SIFT_create(500)
        orb = cv2.ORB_create(100)


        # 获取关键点和描述符
        # kp是一个关键点列表
        # des是一个numpy数组，其大小是关键点数目乘以128,描述符，描述关键点的位置，尺度，向量等信息
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        m = cv2.BFMatcher(crossCheck=True)
        matches = m.match(des1, des2)
        #matches = list(matches)
        matches = sorted(matches,key=lambda x:x.distance)


        inlier = 0
        total_point = 0
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
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
        print(inlier/total_point)
        # sift_res = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), kp1, cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), kp2, matches[:20],None,flags=2)
        #
        # cv2.imwrite("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\dataset3\\" + "19-20.sift.jpg", sift_res)
        # print("len(kp1)",len(kp1))
        # print("len(kp2)", len(kp2))
        # if des1 is None:
        #     print("未检测到sift特征点")
        # else:
        #     #根据sift特征点提取patch，并保存特征点位置以便画图
        #     patches = extract_patches(kp1,img1,65, 10, "cv2")
        #     print(len(patches))
        #     #将特征点位置按行存入TXT文件，每行之间以空格分隔坐标值
        #     txt_index_path = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000019index.txt"
        #     file = open(txt_index_path, "w")
        #
        #     for j in range(0, len(patches)):
        #         #保存patch
        #         path = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000019\\" + str(j) + ".jpg"
        #         cv2.imwrite(path, patches[j])
        #         if j == len(patches) - 1:
        #             file.write("%s" % str(kp1[j].pt[0]) + " " + str(kp1[j].pt[1]))
        #         else:
        #             file.write((
        #                 "%s" % str(kp1[j].pt[0]) + " " + str(kp1[j].pt[1])) + '\n')
        #
        #     patches_2 = extract_patches(kp2, img2, 65, 10, "cv2")
        #     print("len(patches_2)", len(patches_2))
        #
        #     txt_index_path_2 = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000020index.txt"
        #     file = open(txt_index_path_2, "w")
        #
        #     for j in range(0, len(patches_2)):
        #         path = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3\\00000020\\" + str(j) + ".jpg"
        #         cv2.imwrite(path, patches_2[j])
        #         if j == len(patches_2) - 1:
        #             file.write("%s" % str(kp2[j].pt[0]) + " " + str(kp2[j].pt[1]))
        #         else:
        #             file.write((
        #                 "%s" % str(kp2[j].pt[0]) + " " + str(kp2[j].pt[1])) + '\n')

                # 在图中显示特征点
            # img_small_sift = cv2.drawKeypoints(img1, kp1, outImage=np.array([]),
            #                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # img_big_sift = cv2.drawKeypoints(img2, kp2, outImage=np.array([]),
            #                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv2.imwrite("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3" + "1.jpg", img_small_sift)
            # cv2.imwrite("D:\\graduation design\\SIMCLR-pytorch\\simclr\\data\\patches\\dataset3" + "2.jpg", img_big_sift)
            # plt.show()
