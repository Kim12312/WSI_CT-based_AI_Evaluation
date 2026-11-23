# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

case_ids1=['150190']

for case_id in case_ids1:
    print(case_id)
    ctdata=np.load("./ct_visualization_data/"+str(case_id)+"_ct_data.npy")
    ctdata = ctdata.reshape((150,150))
    ctdata /= np.max(ctdata)
    heatmap=np.load("./ct_visualization_data/"+str(case_id)+"_ct_heatmap.npy")
    
    plt.figure(figsize=(10, 8))

    # 显示原始医疗图像（灰度）
    plt.imshow(ctdata, 
               cmap='gray', 
               vmin=np.percentile(ctdata, 5),
               vmax=np.percentile(ctdata, 95))
    
    im = plt.imshow(heatmap, 
                    cmap='jet', 
                    alpha=0.4,
                    vmin=0,
                    vmax=1)     
    
    plt.axis('off')
    plt.savefig('./Figure1/'+str(case_id) + '_heatmap1_ct.png', dpi=500, bbox_inches='tight')
    #plt.savefig('medical_heatmap.png', dpi=150, bbox_inches='tight')
    #plt.show()
    
    ctdata1 = cv2.applyColorMap(np.uint8(255 * ctdata), cv2.COLORMAP_BONE)
    
    plt.figure()
    plt.imshow(ctdata, 'gray')
    plt.axis('off')
    plt.savefig('./Figure1/'+str(case_id) + '_ori_img_ct.png', dpi=500, bbox_inches='tight')
    