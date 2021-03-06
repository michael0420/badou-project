"""
设原图像的尺寸M*M，目标图像的尺寸N*N。
目标图像在原图像的位置是(x,y)
原图像的坐标（xm,ym）,m=0,...,M-1
目标图像的坐标(xn,yn),n=0,...,N-1
原图像的几何中心(x(M-1)/2,y(M-1)/2),目标图像的几何中心(x(N-1)/2,y(N-1)/2)
缩放比例要一致：M/N = (M-1)/2 / (N-1)/2
故 M-1/2 +Z = ((N-1)/2 +Z)M/N
M-1 / 2 +Z = (N-1)M/2N + M/N *Z
N-M / N * Z = N - M / 2N
Z = 1 / 2



"""