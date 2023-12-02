import cv2
from scipy.spatial.distance import cosine
import numpy as np
from gaborpy import gaborcls
# 对图像用kmeans聚类
# 显示图片的函数
def show(winname,src):
    cv2.namedWindow(winname,cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(winname,src)
    cv2.waitKey()

img = cv2.imread('mosaic B.bmp')
o = img.copy()
#print(img.shape):(256, 256, 3)
# 将一个像素点的rgb值作为一个单元处理，这一点很重要
data = img.reshape((-1,3))
# print(data.shape):(65536, 3)
# 转换数据类型
data = np.float32(data)

# 设置Kmeans参数
critera = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,0.1)
flags = cv2.KMEANS_RANDOM_CENTERS
# 对图片进行i分类
data2=data.copy()
for i in range(2,6):
    print(i)
    r, best, center = cv2.kmeans(data2, i, None, criteria=critera, attempts=10, flags=flags)
    # criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
    # print(r)
    # print(best.shape)
    #print(center)
    center = np.uint8(center)
    # 将不同分类的数据重新赋予另外一种颜色，实现分割图片
    data[best.ravel() == 0] = (0, 0, 0)  # 黑色
    data[best.ravel() == 1] = (255, 0, 0)  # 红色
    data[best.ravel() == 2] = (0, 0, 255)  # 蓝色
    data[best.ravel() == 3] = (0, 255, 0)  # 绿色
    data[best.ravel() == 4] = (255, 255, 255)  # 白色
    # 将结果转换为图片需要的格式
    data = np.uint8(data)
    oi = data.reshape((img.shape))
    cv2.imwrite(str(i) + '.jpg', oi)

fea=gaborcls('mapB.bmp')
print(fea)
min_num=float("inf")
flag=0
for i in range(2,6):
    filename = str(i) + '.jpg'
    newfea = gaborcls(filename)
    print(newfea)
    tmp=cosine(newfea,fea)#余弦相似性
    if(min_num>tmp):
        min_num=tmp
        flag=i
        print(flag)
filename = str(flag) + '.jpg'
# 显示图片
oi=cv2.imread(filename)
show('img',img)
show('res',oi)