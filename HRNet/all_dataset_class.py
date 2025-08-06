import os
import numpy as np
import skimage.exposure as sk_exposure
import skimage.io as sk_io
import skimage.transform as sk_transform
from torch.utils.data import Dataset

class BaseLandmarkDataset(Dataset):
    def __init__(self, load_image_dir, load_annotation_dir, target_size, num_landmarks, 
                 use_template=False, heatmap_sigma=3, heatmap_radius=50, heatmap_downsample=1,
                 do_augmentation=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_image_dir = load_image_dir
        self.load_annotation_dir = load_annotation_dir
        self.load_filename_list = os.listdir(load_image_dir)
        self.template_file_path = os.path.join(load_annotation_dir, 'template.txt')
        self.do_augmentation = do_augmentation
        
        # 子类特定参数
        self.target_size = target_size  # (H, W)
        self.num_landmarks = num_landmarks
        self.use_template = use_template
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_radius = heatmap_radius
        self.heatmap_downsample = heatmap_downsample

    def load_image(self, image_file_path):
        image = sk_io.imread(image_file_path)
        if np.size(image.shape) != 3:
            image = np.dstack([image] * 3)
        return image

    def augment(self, image, landmark_list, center, out_shape):
        # 旋转增强
        if np.random.random() < 0.4:
            angle = int(np.random.randint(-20, 20, 1))
            rotate_image = sk_transform.rotate(image, angle, resize=True)
            rotation_matrix = np.array([
                [np.cos(np.radians(angle)), np.sin(np.radians(angle))],
                [-np.sin(np.radians(angle)), np.cos(np.radians(angle))]
            ])
            for i in range(len(landmark_list)):
                lm = np.array(landmark_list[i])
                new_lm = np.dot(rotation_matrix, lm - center) + center
                landmark_list[i] = [new_lm[0], new_lm[1]]
            
            # 裁剪回原始尺寸
            delta_y = (rotate_image.shape[0] - out_shape[0]) // 2
            delta_x = (rotate_image.shape[1] - out_shape[1]) // 2
            image = rotate_image[delta_y:delta_y+out_shape[0], delta_x:delta_x+out_shape[1], :]
        
        # 缩放增强
        if np.random.random() < 0.4:
            scale = np.random.randint(80, 120) / 100
            scale_image = sk_transform.rescale(image, [scale, scale, 1])
            for i in range(len(landmark_list)):
                lm = np.array(landmark_list[i])
                new_lm = (lm - center) * scale + center
                landmark_list[i] = [new_lm[0], new_lm[1]]
                
            if scale >= 1:
                delta_y = (scale_image.shape[0] - out_shape[0]) // 2
                delta_x = (scale_image.shape[1] - out_shape[1]) // 2
                image = scale_image[delta_y:delta_y+out_shape[0], delta_x:delta_x+out_shape[1], :]
            else:
                delta_y = (out_shape[0] - scale_image.shape[0]) // 2
                delta_x = (out_shape[1] - scale_image.shape[1]) // 2
                zero_image = np.zeros((*out_shape, 3))
                zero_image[delta_y:delta_y+scale_image.shape[0], delta_x:delta_x+scale_image.shape[1], :] = scale_image
                image = zero_image
        
        # 对比度变换
        if np.random.random() < 0.4:
            gamma = np.random.randint(80, 120) / 100
            image = sk_exposure.adjust_gamma(image, gamma)
            
        return image, landmark_list

    def generate_gaussian_heatmap(self, landmark_list, shape):
        """
        生成高斯热图并进行下采样
        :param landmark_list: 关键点坐标列表 [[x1, y1], [x2, y2], ...]
        :param shape: 原始热图形状 (num_landmarks, H, W)
        :return: 下采样后的热图
        """
        # 创建原始热图
        original_heatmap = np.zeros(shape)

        # 为每个关键点生成高斯分布
        for i, point in enumerate(landmark_list):
            # 获取坐标点 (x, y)
            x0, y0 = point[0], point[1]
            x0, y0 = int(x0), int(y0)
            radius = self.heatmap_radius
            diameter = 2 * radius + 1
            sigma = self.heatmap_sigma
            
            # 创建高斯核
            m, n = [(ss - 1) / 2 for ss in (diameter, diameter)]
            xx, yy = np.ogrid[-m:m+1, -n:n+1]
            h = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            
            # 获取热图尺寸 (H, W)
            heatmap_h, heatmap_w = shape[1], shape[2]
            
            # 计算边界
            x_min = max(0, x0 - radius)
            x_max = min(heatmap_w, x0 + radius + 1)
            y_min = max(0, y0 - radius)
            y_max = min(heatmap_h, y0 + radius + 1)
            
            # 跳过无效区域
            if x_min >= x_max or y_min >= y_max:
                continue
                
            # 计算高斯核裁剪区域
            gauss_x_min = radius - (x0 - x_min)
            gauss_x_max = radius + (x_max - x0)
            gauss_y_min = radius - (y0 - y_min)
            gauss_y_max = radius + (y_max - y0)
            
            # 裁剪高斯核
            cropped_gaussian = h[
                int(gauss_y_min):int(gauss_y_max),
                int(gauss_x_min):int(gauss_x_max)
            ]
            
            # 将高斯核应用到热图上
            region = original_heatmap[i, y_min:y_max, x_min:x_max]
            if region.shape == cropped_gaussian.shape:
                np.maximum(region, cropped_gaussian, out=region)

        # 应用下采样
        downsample = self.heatmap_downsample

        # 检查下采样参数是否有效
        if downsample > 1:
            if downsample % 2 != 0:
                raise ValueError(f"Invalid downsample value {downsample}. Downsample must be an even number when greater than 1.")
            
            # 计算下采样后的尺寸
            num_landmarks = shape[0]
            downsampled_h = shape[1] // downsample
            downsampled_w = shape[2] // downsample
            
            # 创建下采样热图
            downsampled_heatmap = np.zeros((num_landmarks, downsampled_h, downsampled_w))
            
            # 执行下采样 - 使用最大值池化
            for i in range(num_landmarks):
                for y in range(downsampled_h):
                    for x in range(downsampled_w):
                        # 提取池化区域
                        patch = original_heatmap[i, 
                                                y*downsample:(y+1)*downsample,
                                                x*downsample:(x+1)*downsample]
                        # 取区域内的最大值
                        downsampled_heatmap[i, y, x] = np.max(patch)
            
            return downsampled_heatmap

        # 如果下采样为1，直接返回原始热图
        return original_heatmap

    def load_landmarks(self, annotation_file_path):
        landmark_list = []
        #如果注释文件不存在，返回空列表
        if not os.path.exists(annotation_file_path):
            # print(f"注释文件 {annotation_file_path} 不存在，返回空列表")
            return landmark_list
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:self.num_landmarks]:
                coords = [int(x) for x in line.strip().split(',')]
                landmark_list.append(coords)
        return landmark_list

    def load_template(self):
        if not self.use_template:
            return None
            
        template_list = []
        with open(self.template_file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:self.num_landmarks]:
                coords = [float(x) for x in line.strip().split(',')]
                coords= [coords[0]*self.target_size[0], coords[1]*self.target_size[1]]  # 确保是二维坐标
                template_list.append(coords)
        return template_list

    def _normalize_points(self, points, w, h):
        # 以图像中心为原点(-1,1)
        return [[p[0]/(w-1)*2-1, p[1]/(h-1)*2-1] for p in points]

    def __getitem__(self, index):
        # 获取文件路径
        filename = self.load_filename_list[index]
        img_path = os.path.join(self.load_image_dir, filename)
        ann_path = os.path.join(self.load_annotation_dir, 
                               os.path.splitext(filename)[0] + '.txt')
        
        # 加载图像和标注
        image = self.load_image(img_path)
        orig_h, orig_w = image.shape[:2]
        landmarks = self.load_landmarks(ann_path)
        
        # 加载模板（如果使用）
        template = self.load_template() if self.use_template else None
        
        # 图像缩放
        image = sk_transform.resize(image, self.target_size)
        scale_x = self.target_size[1] / orig_w  # 宽度缩放比例
        scale_y = self.target_size[0] / orig_h  # 高度缩放比例
        
        # 缩放标注点
        scaled_landmarks = []
        for lm in landmarks:
            scaled_landmarks.append([lm[0] * scale_x, lm[1] * scale_y])
        
        # # 缩放模板点
        # scaled_template = None
        # if template:
        #     scaled_template = []
        #     for tmpl in template:
        #         scaled_template.append([tmpl[0] * scale_x, tmpl[1] * scale_y])
        scaled_template=template
        
        # 数据增强（只对图像和标注点进行）
        center = np.array([self.target_size[1]//2, self.target_size[0]//2])  # (x, y)
        if self.do_augmentation:
            image, scaled_landmarks = self.augment(
                image, scaled_landmarks, center, self.target_size
            )
        
        # 生成热图
        heatmap = self.generate_gaussian_heatmap(
            scaled_landmarks, 
            (self.num_landmarks, self.target_size[0], self.target_size[1])
        )
        
        # 转置图像通道
        image = np.transpose(image, (2, 0, 1))
        
        # 归一化坐标点
        normalized_landmarks = None
        normalized_template = None
        if scaled_landmarks:
            normalized_landmarks = np.asarray(self._normalize_points(
                scaled_landmarks, self.target_size[1], self.target_size[0]))  # w, h
        if scaled_template:
            normalized_template = np.asarray(self._normalize_points(
                scaled_template, self.target_size[1], self.target_size[0]))  # w, h
        
        # 根据子类需求返回不同结构
        if self.use_template:
            return filename, image, heatmap, normalized_landmarks, normalized_template
        else:
            return filename, image, heatmap

    def __len__(self):
        return len(self.load_filename_list)



