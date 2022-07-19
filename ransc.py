import csv
from utils.hpatch import *
import time
import os


def drawMatches(img1, des1, img2, des2, matches):
    """
    author：sorry to not cite because i forget,i will update soon
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
    rows1, cols1, rows2, cols2 = img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype="uint8")

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for i in range(len(des1)):
        (x1,y1) = des1[i][0],des1[i][1]
        (x2,y2) = des2[i][0],des2[i][1]
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        a, b, c = np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)
        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (a, b, c), 1)      #画圆，cv2.circle()参考官方文档
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (a, b, c), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (a, b, c), 1, lineType=cv2.LINE_AA, shift=0)  #画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out


if __name__ == '__main__':
    #加载记录两幅图像对应的特征向量文件
    des1 = np.loadtxt("dataset3\\00000019.csv", delimiter=';')  # 打开csv文件
    des2 = np.loadtxt("dataset3\\00000020.csv", delimiter=';')  # 打开csv文件

    img1 = cv2.imread("Images\\00000019.png")
    img2 = cv2.imread("Images\\00000020.png")

    if len(des1) > 1 and len(des1) != 2048:  #一点小问题，当只有一个特征变量时，读取csv文件会得到2048维
        # 定义FLANN匹配器------KD树，具体原理尚不清楚，index_params中的algorithm为0或1
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # 匹配
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        m = cv2.BFMatcher(crossCheck=True)
        matches = m.match(des1, des2)

        #读取记录特征点位置的文件
        txt_index1_path = "dataset3\\00000019index.txt"
        file = open(txt_index1_path)
        index_1 = file.read().split("\n")

        txt_index2_path = "dataset3\\00000020index.txt"
        file2 = open(txt_index2_path)
        index_2 = file2.read().split("\n")

        src_pts = []
        dst_pts = []
        good = []
        #得到匹配点的坐标
        for m in matches:  #
            good.append(m)
            src_x, src_y = index_1[m.queryIdx].split()
            dst_x, dst_y = index_2[m.trainIdx].split()
            src_pts.append([np.float32(src_x), np.float32(src_y)])
            dst_pts.append([np.float32(dst_x), np.float32(dst_y)])

        src_pts_final = []
        dst_pts_final = []

        if len(good) > 4:
            # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
            src_pts = np.int32(src_pts)
            dst_pts = np.int32(dst_pts)
            # findHomography 函数是计算变换矩阵
            # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
            # 返回值：M 为变换矩阵，mask是掩模
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # ravel方法将数据降维处理，最后并转换成列表格式
            matchesMask = mask.ravel().tolist()
            for m in range(len(matchesMask)):
                if matchesMask[m] == 1:
                    src_pts_final.append(src_pts[m])
                    dst_pts_final.append(dst_pts[m])
            # 获取img1的图像尺寸
            h, w, dim = img1.shape
            # pts是图像img1的四个顶点
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # 计算变换后的四个顶点坐标位置
            if M is None:
                print("M is None")
            else:
                dst = cv2.perspectiveTransform(pts, M)
                out = drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), src_pts_final,
                                  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), dst_pts_final, matchesMask)
                cv2.imwrite("dataset3\\" + "19-20.simclr.jpg", out)




