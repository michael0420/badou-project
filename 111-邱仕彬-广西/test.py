'''
需求：1、图片转灰度图
           2、图片转黑白二值图
'''

from PIL import Image

window = Image.new('RGB', (1200, 600), 'white')

# 灰度图
def image_gary():
    # name = Image.open('lenna.png').convert('L')
    # name.save('test1.jpg')
    # or
    img = Image.open('lenna.png').convert('RGB')
    for i in range(img.width):
        for j in range(img.height):
            R, G, B = img.getpixel((i, j))
            gary = int(R*0.3 + G*0.59 + B*0.11)
            window.putpixel((i+68, j+44), (gary,)*3)

# 二值图
def image_double():
    img = Image.open('lenna.png').convert('RGB')
    for i in range(img.width):
        for j in range(img.height):
            R, G, B = img.getpixel((i, j))
            WB = (0, 0, 0) if (R+G+B)/(255*3) < 0.5 else (255, 255, 255)
            window.putpixel((i+648, j+44), WB)

if __name__ == '__main__':
    image_gary()
    image_double()
    window.save('test.jpg')