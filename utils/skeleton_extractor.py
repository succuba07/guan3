# utils/skeleton_extractor.py
import cv2
import numpy as np
from PIL import Image

class SkeletonExtractor:
    @staticmethod
    def extract(image):
        """
        从输入图像（PIL Image）中提取骨架
        返回：3通道骨架图像（PIL Image）
        """
        # 转为OpenCV格式（RGB→BGR）
        img_np = np.array(image)
        height, width = img_np.shape[:2]  # 关键修复：定义height和width变量
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 转为灰度图
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测（Canny）
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学细化（提取骨架）
        skeleton = cv2.ximgproc.thinning(edges)
        
        # 转为3通道（与输入图像通道数匹配）
        skeleton_3ch = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
         # 确保输出尺寸与输入相同
        if skeleton_3ch.shape[0] != height or skeleton_3ch.shape[1] != width:
            skeleton_3ch = cv2.resize(skeleton_3ch, (width, height))
        
        # 转回PIL Image
        return Image.fromarray(skeleton_3ch)
        # 转回PIL Image并包装成列表
        # return [Image.fromarray(skeleton_3ch)]
