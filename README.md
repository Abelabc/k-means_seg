# kmeans算法进行图像分割
![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)
## 任务要求
使用opencv的K-means函数对图像进行先分割，控制迭代目标停止，误差小于0.1。其次，由于k值的不确定性，利用skimage.gabor滤波得到图像特征，为了量化分割图像和实际分割图像的似然性，通过gabor滤波得到列向量，然后对两幅图像的列向量进行余弦相似度计算，得到与实际分割图像最接近的分割图像;
```
    # 将不同分类的数据重新赋予另外一种颜色，实现分割图片
    data[best.ravel() == 0] = (0, 0, 0)  # 黑色
    data[best.ravel() == 1] = (255, 0, 0)  # 红色
    data[best.ravel() == 2] = (0, 0, 255)  # 蓝色
    data[best.ravel() == 3] = (0, 255, 0)  # 绿色
    data[best.ravel() == 4] = (255, 255, 255)  # 白色
```
[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/2.jpg)](https://github.com/Abelabc/k-means_seg/blob/main/pic/2.jpg "K=2")

> 图为：K=2

[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/3.jpg)](https://github.com/Abelabc/k-means_seg/blob/main/pic/3.jpg "K=3")

> 图为：K=3

[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/4.jpg)](https://github.com/Abelabc/k-means_seg/blob/main/pic/4.jpg "K=4")

> 图为：K=4

[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/5.jpg)](https://github.com/Abelabc/k-means_seg/blob/main/pic/5.jpg "K=5")

> 图为：K=5


## 通过定义 gaborcls 函数计算原始 mapB.bmp 的特征值
```
def gaborcls(filename):
    img = cv2.imread(filename)  # 读图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
    frequency = 0.6
    # gabor变换
    real, imag = filters.gabor(img_gray, frequency=0.6, theta=60, n_stds=5)
    # 取模
    img_mod = np.sqrt(real.astype(float) ** 2 + imag.astype(float) ** 2)
    # 图像缩放（下采样）
    newimg = cv2.resize(img_mod, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_AREA)
    tempfea = newimg.flatten()  # 矩阵展平
    tmean = np.mean(tempfea)  # 求均值
    tstd = np.std(tempfea)  # 求方差
    newfea = (tempfea - tmean) / tstd  # 数值归一化
    return newfea
```
## The most similar image obtained in the end is 3.jpg：

[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/3.jpg)](https://github.com/Abelabc/k-means_seg/blob/main/pic/3.jpg "K=3")

> 图为：K=3

[![](https://github.com/Abelabc/k-means_seg/blob/main/pic/mapB.bmp)](https://github.com/Abelabc/k-means_seg/blob/main/pic/mapB.bmp "map")

>图为：分割图
