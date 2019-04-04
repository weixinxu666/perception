#！／user/bin/env python
#-*- coding:utf-8 -*-
import copy
from matplotlib import pyplot as plt
from matplotlib import animation#绘制动图

training_set=[[[3,3],1],[[4,2],1],[[1,1],-1],[[2,0],-1]]

#初值选择为0
w=[0,0]
b=0
alpha = 1
history=[]
#对应于（3）

def update(item):
   # 更新参数，步长(学习率)为1
   global w,b,history
   w[0]+=alpha*item[1]*item[0][0]
   w[1]+=alpha*item[1]*item[0][1]
   b+=alpha*item[1]
   print(w,b)
   history.append([copy.copy(w),b])


def loss(item):#计算损失函数值y(w*x+b)
    res=0
    for i in range(len(item[0])):
        res+=item[0][i]*w[i]
    res+=b
    res*=item[1]
    # print(res)
    return res


def check():
    #检查是否需要继续更新
    flag=False
    for item in training_set:
        # print(item)
        if loss(item)<=0:
            flag=True
            update(item)
    if not flag:
        print("RESULT:w:"+str(w)+"  b:"+str(b))
    return flag


if __name__=="__main__":
    for i in range(1000):
        if not check():
            break

#下面是绘图
fig=plt.figure()
ax=plt.axes(xlim=(0,2),ylim=(-2,2))
line,=ax.plot([],[],'g',lw=2)
label=ax.text([],[],'')


def init():
    line.set_data([],[])
    x,y,x_,y_,=[],[],[],[]
    for p in training_set:
        if p[1]>0:
            x.append(p[0][0])
            y.append(p[0][1])
        else:
            x_.append(p[0][0])
            y_.append(p[0][1])
    plt.plot(x,y,'bo',x_,y_,'r^')
    plt.axis([-6,6,-6,6])
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Perception')
    return line,label


def animate(i):
    global history,ax,line,label
    w=history[i][0]
    b=history[i][1]
    if w[1]==0:return line,label
    x1 = -7.0
    y1 = -(b + w[0] * x1) / w[1]
    x2 = 7.0
    y2 = -(b + w[0] * x2) / w[1]
    line.set_data([x1, x2], [y1, y2])
    x1 = 0.0
    y1 = -(b + w[0] * x1) / w[1]
    label.set_text(str(history[i][0]) + ' ' + str(b))
    label.set_position([x1, y1])
    return line, label

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=True,blit=True)
#保存为动图需要安装imagemagick模块
anim.save('sin_dot.gif', writer='imagemagick', fps=60)
plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# p_x = np.array([[3, 3], [4, 3], [1, 1]])
# y = np.array([1, 1, -1])
# plt.figure()
# for i in range(len(p_x)):
#     if y[i] == 1:
#         plt.plot(p_x[i][0], p_x[i][1], 'ro')
#     else:
#         plt.plot(p_x[i][0], p_x[i][1], 'bo')
#
# w = np.array([1, 0])
# b = 0
# delta = 1
#
# for i in range(100):
#     choice = -1
#     for j in range(len(p_x)):
#         if y[j] != np.sign(np.dot(w, p_x[0]) + b):
#             choice = j
#             break
#     if choice == -1:
#         break
#     w = w + delta * y[choice] * p_x[choice]
#     print("weight:",w)
#     b = b + delta * y[choice]
#     print("bias：",b)
#
# line_x = [0, 10]
# line_y = [0, 0]
#
# for i in range(len(line_x)):
#     line_y[i] = (-w[0] * line_x[i] - b) / w[1]
#
# plt.plot(line_x, line_y)
# plt.savefig("picture.png")