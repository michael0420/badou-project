# coding=utf8

import tkinter
from tkinter.messagebox import *
from tkinter import *

import cv2
import numpy as np

img = cv2.imread('lena.jpg')
h = 500
w = 250
poly = []
window = None

class MyDialog(Toplevel):
    # 定义构造方法
    def __init__(self, parent, title = None, modal=True, msg="None"):
        Toplevel.__init__(self, parent)
        self.transient(parent)
        # 设置标题
        # if title:
        #     self.title(title)
        #
        self.parent = parent
        # self.result = None

        self.msg = msg

        # 创建对话框的主体内容
        frame = Frame(self)

        # 调用init_widgets方法来初始化对话框界面
        self.initial_focus = self.init_widgets(frame)
        frame.pack(padx=5, pady=5)

        # 调用init_buttons方法初始化对话框下方的按钮
        # self.init_buttons()

        # 根据modal选项设置是否为模式对话框
        # if modal:
        #     self.grab_set()

        if not self.initial_focus:
            self.initial_focus = self

        # 为"WM_DELETE_WINDOW"协议使用self.cancel_click事件处理方法
        self.protocol("WM_DELETE_WINDOW", self.cancel_click)

        # 根据父窗口来设置对话框的位置
        # self.geometry("+%d+%d" % (parent.winfo_rootx()+50, parent.winfo_rooty()+50))
        self.geometry("+%d+%d" % (250, 450))
        # print( self.initial_focus)

        # 让对话框获取焦点
        # self.initial_focus.focus_set()
        # self.wait_window(self)

    # 通过该方法来创建自定义对话框的内容
    def init_widgets(self, master):
        # 创建并添加Label
        Label(master, text=self.msg, font=12).grid(row=1, column=0)

        # 创建并添加Entry,用于接受用户输入的用户名
        # self.name_entry = Entry(master, font=16)
        # self.name_entry.grid(row=1, column=1)

        # 创建并添加Label
        # Label(master, text='密 码', font=12,width=10).grid(row=2, column=0)

        # 创建并添加Entry,用于接受用户输入的密码
        # self.pass_entry = Entry(master, font=16)
        # self.pass_entry.grid(row=2, column=1)

        # 通过该方法来创建对话框下方的按钮框
    def init_buttons(self):
        f = Frame(self)

        # 创建"确定"按钮,位置绑定self.ok_click处理方法
        w = Button(f, text="确定", width=10, command=self.ok_click, default=ACTIVE)
        w.pack(side=LEFT, padx=5, pady=5)

        # 创建"确定"按钮,位置绑定self.cancel_click处理方法
        w = Button(f, text="取消", width=10, command=self.cancel_click)
        w.pack(side=LEFT, padx=5, pady=5)
        self.bind("", self.ok_click)
        self.bind("", self.cancel_click)
        f.pack()

    # 该方法可对用户输入的数据进行校验
    def validate(self):
        # 可重写该方法
        return True

    # 该方法可处理用户输入的数据
    def process_input(self):
        user_name = self.name_entry.get()
        user_pass = self.pass_entry.get()
        messagebox.showinfo(message='用户输入的用户名: %s, 密码: %s' % (user_name , user_pass))

    def ok_click(self, event=None):
        print('确定')

        # 如果不能通过校验，让用户重新输入
        if not self.validate():
            self.initial_focus.focus_set()
            return self.withdraw()
        self.update_idletasks()

        # 获取用户输入数据
        self.process_input()

        # 将焦点返回给父窗口
        self.parent.focus_set()

        # 销毁自己
        self.destroy()

    def cancel_click(self, event=None):
        print('取消')

        # 将焦点返回给父窗口
        # self.parent.focus_set()
        # 销毁自己
        self.destroy()

def showmsg(parent, msg):
    # window = tkinter.Tk()  # *********
    # window.withdraw()  # ****实现主窗口隐藏
    # window.update()  # *********需要update一下

    MyDialog(parent=parent, msg=msg, modal=False)
    # window.destroy()

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(x, y)
        cv2.circle(img, (x, y), 2, (0, 0, 255))
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255))
        # cv2.imshow("image", img)

        poly.append([x, y])
        if len(poly) == 4:
            perspective()


def main():
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    i = 0
    while (1):
        cv2.imshow("image", img)
        title = None
        if len(poly) == i:
            i = i + 1
            if i == 1:
                title = "请选择左上角"
            if i == 2:
                title = "请选择右上角"
            if i == 3:
                title = "请选择左下角"
            if i == 4:
                title = "请选择右下角"

            showinfo('', title, parent=None)
            # showmsg(parent=window, msg=title)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if len(poly) == 4:
            break
    cv2.destroyAllWindows()


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


def perspective():
    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32(poly)
    dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    print(img.shape)
    # 生成透视变换矩阵；进行透视变换
    m = cv2.getPerspectiveTransform(src, dst)
    # m = WarpPerspectiveMatrix(src, dst)
    print("warpMatrix:")
    print(m)
    img1 = img.copy()
    result = cv2.warpPerspective(img1, m, (w, h))
    # cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

def message_askyesno():
    window = tkinter.Tk()  # *********
    window.withdraw()  # ****实现主窗口隐藏
    window.update()  # *********需要update一下

    txt = tkinter.messagebox.askyesno("提示","要执行此操作？")
    window.destroy()

    return txt


if __name__ == "__main__":
    # showmsg("hello world")

    window = tkinter.Tk()
    window.withdraw()  # 退出默认 tk 窗口
    window.update()

    main()
