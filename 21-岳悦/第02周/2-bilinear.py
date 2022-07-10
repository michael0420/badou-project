import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import argparse
import shutil
import copy

def bilinear(srcimg,dstheight,dstwidth):
    srcheight,srcwidth,channels = srcimg.shape
    dstimage = np.zeros((dstheight,dstwidth,channels),dtype=np.uint8)
    rate_width = float(srcwidth) / dstwidth
    rate_height = float(srcheight) / dstheight
    if srcheight == dstheight and srcwidth == dstwidth:
        dstimage = copy.deepcopy(srcimg)
    else:
        for channel in range(channels):
            for i in range(dstheight):
                for j in range(dstwidth):
                    # x为纵向，y为横向
                    # opencv对应：左上角是原点，往下是x/height。往右是y/width。
                    # 双线性插值对应：左上角是原点，往上是y/height。往右是x/width。
                    src_y = int(i*rate_height)
                    src_x = int(j*rate_width)

                    src_x1 = int(np.floor(src_x))
                    src_y1 = int(np.floor(src_y))
                    src_x2 = min(src_x1 + 1 ,srcwidth - 1)
                    src_y2 = min(src_y1 + 1, srcheight - 1)

                    src_r1 = (src_x2 - src_x) * srcimg[src_y1,src_x1,channel] + (src_x - src_x1) * srcimg[src_y1,src_x2,channel]
                    src_r2 = (src_x2 - src_x) * srcimg[src_y2,src_x1,channel] + (src_x - src_x1) * srcimg[src_y2,src_x2,channel]
                    dstimage[i,j,channel] = int((src_y2-src_y)*src_r1+(src_y-src_y1)*src_r2)

    return dstimage

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please input height and width'
    parser.add_argument("--height",dest="height",default=320,type=np.uint8)
    parser.add_argument("--width",dest="width",default=320,type=np.uint8)
    args = parser.parse_args()

    imagepath = os.getcwd()+'/image/lenna.png'
    originimage = cv2.imread(imagepath)
    # srcimage = cv2.cvtColor(originimage,cv2.COLOR_BGR2RGB)
    srcimage = originimage
    print(args.height,args.width)
    outpath = os.getcwd()+'/result/bilinear/'
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath)
    cv2.imwrite(outpath+'/src.jpg',srcimage)
    print(srcimage.shape)
    dstimage = bilinear(srcimage,args.height,args.width)
    # plt.figure()
    # plt.imshow(dstimage)
    # plt.show()
    # dstimage = cv2.cvtColor(dstimage,cv2.COLOR_BGR2RGB)
    cv2.imwrite(outpath+'/bilinear.jpg',dstimage)
    print(dstimage.shape)
    cv2.waitKey(0)
