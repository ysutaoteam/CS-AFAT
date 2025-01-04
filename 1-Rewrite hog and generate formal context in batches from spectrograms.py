class Hog_descriptor():
    '''
    计算能量点的梯度信息
    '''

    def __init__(self, img, cell_size=8, bin_size=9):

        self.img = img
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))
        self.img = self.img * 255
        # print('img',self.img.dtype)   #float64
        # 参数初始化
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 180 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 180 % self.bin_size == 0, "bin_size should be divisible by 180"

    def extract(self):
        '''
        计算图像的HOG描述符，以及HOG-image特征图
        '''

        height, width = self.img.shape

        '''
        1、计算图像每一个像素点的梯度幅值和角度
        '''
        gradient_magnitude, gradient_angle = self.global_gradient()  #返回每一像素点的梯度大小和角度值
        #梯度取绝对值
        gradient_magnitude = abs(gradient_magnitude)

        '''
        2、计算输入图像的每个cell单元的梯度直方图
        '''
        m = 0
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        # 遍历每一行、每一列
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                # 计算第[i][j]个cell的特征向量    每一个[i][j]中存8*8个像素点的梯度值和角度值
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                m = m + 1   # 在cell_gradient中在传一个参数m,用于cell_gradient中显示第几个细胞
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle,m)
        print(m)


        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)

        '''
        3、将2x2个cell组成一个block，一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
           将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，
           这就是最终分类的特征向量
        '''
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
        分别计算图像沿x轴和y轴的梯度
        '''
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)      #dx
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)      #dy
        '''
        返回每一个像素点的梯度大小和角度值
        '''
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)

        # 角度大于180°的，减去180度
        gradient_angle[gradient_angle > 180.0] -= 180
        # print('gradient',gradient_magnitude.shape,gradient_angle.shape,np.min(gradient_angle),np.max(gradient_angle))
        return gradient_magnitude, gradient_angle


    def cell_gradient(self, cell_magnitude, cell_angle,m):
        '''
        为每个细胞单元构建梯度方向直方图
        '''
        d2 = {}  #用于存储细胞中所有字典  64个
        kk = 0 #用来计算一个细胞64个像素的遍历
        orientation_centers = [0] * self.bin_size   #构建[0, 0, 0, 0, 0, 0, 0, 0, 0]进行存储
        # 遍历cell中的每一个像素点
        for i in range(cell_magnitude.shape[0]):         #cell_magnitude=（8，8）
            for j in range(cell_magnitude.shape[1]):
                # 梯度幅值
                gradient_strength = cell_magnitude[i][j]
                # 梯度方向
                gradient_angle = cell_angle[i][j]
                # 双线性插值
                min_angle, max_angle, weight = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - weight))
                orientation_centers[max_angle] += (gradient_strength * weight)
                # orientation_centers[min_angle] += 1
                # orientation_centers[max_angle] += 1
                kk = kk + 1
                if kk < 65:
                    #字典的一键多值
                    # 创建一个字典，用于存储第几个像素  计算的min_anfle  max_amgle   dict={key1:[value1,value2]}
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
        # workbook.save('F:/HOG-Picture/set-xingshibeijingshengcheng/jiankangren'+ "./" + filename+'.xls')
        # workbook.save('F:/HOG-Picture/set-xingshibeijingshengcheng/youbingren' + "./" + filename + '.xls')
        # workbook.save('F:/HOG-Picture/set-xingshibeijingshengcheng/test/healthyVoice-jiequ' + "./" + filename + '.xls')
        # workbook.save('C:/Users/z/Desktop/shiyan/2-xingshibeijing/PD/2' + "./" + filename + '.xls')
        workbook.save(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\xingshibeijing\HC' + "/" + filename + '.xls')
        print('第'+str(m)+'个细胞的统计直方图：',orientation_centers)
        return orientation_centers


    def get_closest_bins(self, gradient_angle):
        '''
        计算梯度方向gradient_angle位于哪一个bin中，这里采用的计算方式为双线性插值
        args:
            gradient_angle:角度
        return：
            start,end,weight：起始bin索引，终止bin的索引，end索引对应bin所占权重
        '''
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
        # img = cv2.imread('./yuputu.png')
        print('读取到的图像',img)
        width = 64
        height = 192
        img_copy = img[:, :, ::-1]
        gray_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # 创建workbook
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





