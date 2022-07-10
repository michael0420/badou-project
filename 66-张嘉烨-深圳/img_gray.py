import cv2
import os
import datetime
import matplotlib.pyplot as plt

# ------------ 参数初始化------------------
cap = cv2.VideoCapture()  # 初始化摄像头
CAM_NUM = 0  # 摄像头编号
name_flag = None  # 储存文件名，实际为方法调用时的系统时间


# ------------------以当前系统时间生成文件名-------------
def name_generate():
    # 使用当前系统时间命名截取的图片
    now_flag = datetime.datetime.now()
    month_flag = now_flag.month
    day_flag = now_flag.day
    hour_flag = now_flag.hour
    minute_flag = now_flag.minute
    second_flag = now_flag.second
    name_generate_flag = str(month_flag) + "-" \
                         + str(day_flag) + "." \
                         + str(hour_flag) + "." \
                         + str(minute_flag) + "." \
                         + str(second_flag)
    return name_generate_flag

# -------------读取图像--------------
# 打开摄像头
cap.open(CAM_NUM)
# get a frame
cap_flag, image = cap.read()
# 使用当前系统时间命名截取的图片
name_flag = name_generate()
print('图片名：' + name_flag)

# -----------储存-----------
folder_path = './Capture'
if not os.path.exists(folder_path):
    print('Capture folder not exist, creating directory')
    os.makedirs(folder_path)
cv2.imwrite("./Capture/" + str(name_flag) + ".jpg", image)

# ------------------图像处理--------------
# 转灰度图
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./Capture/" + str(name_flag) + "_gray.jpg", img_gray)
# 转2值图
img_2 = img_gray

# 将像素一个一个换成黑白像素点
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if img_gray[i][j] > 100:
            img_2[i][j] = 0
        else:
            img_2[i][j] = 1

# ----------输出2值图结果-----------------
cv2.imwrite("./Capture/" + str(name_flag) + "_binary.jpg", img_2)
cap.release()

plt.figure("a")
plt.imshow(img_2, cmap='gray')
plt.axis('off')
plt.show()