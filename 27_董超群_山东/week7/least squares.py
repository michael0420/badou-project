import pandas as pd

filepath = 'train_data.csv'
#sales = pd.read_csv(filepath,sep='\s*,\s*',engine="python")
sales = pd.read_csv(filepath,engine="python")       #sep='\s*,\s*'  与默认值sep=','有什么区别？

X = sales['X'].values
Y = sales['Y'].values

#求和变量初始化
s1 = 0      #x*y
s2 = 0      #x
s3 = 0      #y
s4 = 0      #x*x
n = len(X)

#求和
for i in range(n):
    s1 += X[i] * Y[i]
    s2 += X[i]
    s3 += Y[i]
    s4 += X[i] * X[i]

#对k、b偏导，令偏导=0，按照解得的公式带入数据
k = (n * s1 - s2 * s3) / (n * s4 - s2 * s2)
b = s3 / n - k * s2 / n
print("y = {}x + {}".format(k,b))


