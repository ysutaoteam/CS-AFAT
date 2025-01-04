import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import xlwt

class Hog_descriptor():


    def __init__(self, img, cell_size=8, bin_size=9):

        self.img = img
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))
        self.img = self.img * 255
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 180 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 180 % self.bin_size == 0, "bin_size should be divisible by 180"

    def extract(self):
        height, width = self.img.shape

        '''
        1、计算图像每一个像素点的梯度幅值和角度
        '''
        gradient_magnitude, gradient_angle = self.global_gradient()  #返回每一像素点的梯度大小和角度值
        #梯度取绝对值
        gradient_magnitude = abs(gradient_magnitude)

        '''
        2、计算输入图像的每个单元的梯度信息
        '''
        m = 0
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))   #形成一个(94,76.9)大小数组
        # 遍历每一行、每一列
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # 计算第[i][j]个cell的特征向量
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                m = m + 1   # 在cell_gradient中在传一个参数m,用于cell_gradient中显示第几个细胞
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle,m)
        print(m)

        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        jishu = 0
        hog_vector = []   #用来存储每个块的统计信息，两两之间串联关系

        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                # 提取第[i][j]个block的特征向量
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                '''块内归一化梯度直方图，去除光照、阴影等变化，增加鲁棒性'''
                # 计算l2范数
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector) + 1e-5
                # 归一化
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
                jishu =jishu+1
        return np.asarray(hog_vector), hog_image

    def global_gradient(self):
        '''
        返回每一个像素点的梯度大小和角度值
        '''
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)      #dx
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)      #dy
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)
        gradient_angle[gradient_angle > 180.0] -= 180
        return gradient_magnitude, gradient_angle


    def cell_gradient(self, cell_magnitude, cell_angle,m):

        d2 = {}  #用于存储细胞中所有字典
        kk = 0 #用来计算一个细胞64个像素的遍历
        orientation_centers = [0] * self.bin_size
        # 遍历cell中的每一个像素点
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                # 梯度幅值
                gradient_strength = cell_magnitude[i][j]
                # 梯度方向
                gradient_angle = cell_angle[i][j]
                # 双线性插值
                min_angle, max_angle, weight = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - weight))
                orientation_centers[max_angle] += (gradient_strength * weight)

                kk = kk + 1
                if kk < 65:
                    d1 = {}
                    key = kk
                    value = min_angle
                    d1.setdefault(key, []).append(value)
                    value2 = max_angle
                    d1.setdefault(key, []).append(value2)

                else:
                    kk = 0
                d2.update(d1)   #每一个像素点为为一个字典d1，将d1给d2，每次在d2后面更新d1
                '''
                将数据写入到表格中，形成形式背景
                '''

                worksheet.write(kk+(64*(m-1)),0,kk)
                worksheet.write(kk + (64 * (m - 1)),min_angle+1, 1)
                worksheet.write(kk + (64 * (m - 1)),max_angle+1,1)

        workbook.save(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\xingshibeijing\HC' + "/" + filename + '.xls')


        print('第'+str(m)+'个细胞的统计直方图：',orientation_centers)
        return orientation_centers


    def get_closest_bins(self, gradient_angle):

        idx = int(gradient_angle / self.angle_unit)   #angle_unit=20
        mod = gradient_angle % self.angle_unit
        return idx % self.bin_size, (idx + 1) % self.bin_size, mod / self.angle_unit

    def render_gradient(self, image, cell_gradient):

        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        # 遍历每一个cell
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                # 获取第[i][j]个cell的梯度直方图
                cell_grad = cell_gradient[x][y]
                # 归一化
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                # 遍历每一个bin区间
                for magnitude in cell_grad:
                    # 转换为弧度
                    angle_radian = math.radians(angle)
                    # 计算起始坐标和终点坐标，长度为幅值(归一化),幅值越大、绘制的线条越长、越亮
                    x1 = int(x * self.cell_size + cell_width + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


if __name__ == '__main__':
    # 加载图像
    for filename in os.listdir(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\yupu\HC/'):  #'F:/HOG-Picture/HealthyPicture'      PatientPicture   TestPicture healthyVoice-jiequ  pantientvice-jiequ
        img = cv2.imread(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\yupu\HC/' + "./" + filename)

        print('读取到的图像',img)
        width = 64
        height = 192
        img_copy = img[:, :, ::-1]
        gray_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        # 创建workbook（其实就是excel，后来保存一下就行）
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建表
        worksheet = workbook.add_sheet('sheet1')
        # 往单元格内写入内容:写入表头
        worksheet.write(0, 0, label="像素点")
        worksheet.write(0, 1, label="0°-20°")
        worksheet.write(0, 2, label="20°-40°")
        worksheet.write(0, 3, label="40°-60°")
        worksheet.write(0, 4, label="60°-80°")
        worksheet.write(0, 5, label="80°-100°")
        worksheet.write(0, 6, label="100°-120°")
        worksheet.write(0, 7, label="120°-140°")
        worksheet.write(0, 8, label="140°-160°")
        worksheet.write(0, 9, label="160°-180°")
        # 显示原图像
        plt.figure(figsize=(6.4, 2.0 * 3.2))
        plt.subplot(1, 2, 1)
        plt.imshow(img_copy)
        # HOG特征提取
        hog = Hog_descriptor(gray_copy, cell_size=8, bin_size=9)
        hog_vector, hog_image = hog.extract()
        print('hog_vector', hog_vector.shape)
        print(hog_vector)
        print('hog_image', hog_image.shape)
        print(hog_image)