# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

case_ids1=['150190']
cases=os.listdir("./Figure1/")
for index1 in range(len(case_ids1)):
    try:
        if str(case_ids1[index1]) in cases:
            continue
            
        model_file="./Figure1/"
        imgdata2=np.load("./wsi_visualization_data/"+str(case_ids1[index1])+"_test1.npy")
        print('start')
        if not os.path.isdir(model_file):
            os.makedirs(model_file)
        
        gray = (imgdata2 * 255).clip(0, 255).astype(np.uint8)
    
        # 对灰度图进行高斯模糊（核大小越大，平滑越强）
        gray_smoothed = cv2.GaussianBlur(gray, (15, 15), 0)
        
        color_mapped = cv2.applyColorMap(gray_smoothed, cv2.COLORMAP_VIRIDIS)
        
        # 双边滤波参数调整（sigmaColor控制颜色融合强度）
        smoothed = cv2.bilateralFilter(color_mapped, d=9, sigmaColor=75, sigmaSpace=75)
    
        plt.figure()
        cv2.imwrite(model_file+str(case_ids1[index1])+"_heatmap1.png", smoothed)
        
        
    except:
        continue
        
    

