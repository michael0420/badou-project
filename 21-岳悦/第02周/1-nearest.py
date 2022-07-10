import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import argparse
import shutil

def nearest(srcimg,dstheight,dstwidth):
    srcheight,srcwidth,channels = srcimg.shape
    dstimage = np.zeros((dstheight,dstwidth,channels),dtype=np.uint8)
    rate_width = float(dstwidth) / srcwidth
    rate_height = float(dstheight) / srcheight
    for channel in range(channels):
        for i in range(dstheight):
            for j in range(dstwidth):
                height = int(i/rate_height)
                width = int(j/rate_width)
                dstimage[i,j,channel] = srcimage[height,width,channel]

    return dstimage

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please input height and width'
    parser.add_argument("--height",dest="height",default=640,type=np.uint8)
    parser.add_argument("--width",dest="width",default=640,type=np.uint8)
    args = parser.parse_args()

    imagepath = os.getcwd()+'/image/lenna.png'
    originimage = cv2.imread(imagepath)
    # srcimage = cv2.cvtColor(originimage,cv2.COLOR_BGR2RGB)
    srcimage = originimage
    print(args.height,args.width)
    outpath = os.getcwd()+'/result/nearest/'
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath)
    cv2.imwrite(outpath+'/src.jpg',srcimage)
    print(srcimage.shape)
    dstimage = nearest(srcimage,args.height,args.width)
    # plt.figure()
    # plt.imshow(dstimage)
    # plt.show()
    # dstimage = cv2.cvtColor(dstimage,cv2.COLOR_BGR2RGB)
    cv2.imwrite(outpath+'/nearest.jpg',dstimage)
    print(dstimage.shape)
    cv2.waitKey(0)
