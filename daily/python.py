python

在终端输入python后，就能进入解释器

>>>是提示符（prompt），告诉你可以输入指令
如果想要退出，可以输入exit()或者按Ctrl-D

运行python程序,输入一个终端python+.py文件即可

-----------------
ipython

b? # 返回基本信息 / b?? #返回源代码
np.*load*? # 显示Numpy顶级命名空间中含有“load”的所有函数

%run test.py # 假设为当前路径
%timeit np.dot(a, a) 
%matplotlib inline # 直接在jupyter中画图
任何代码在执行时，只要按下“Ctrl-C”，就会应发一个KeyboardInterrupt
绝大部分情况下都将立即停止执行

python中object是没有自带类型，每一个object都有一个明确的类型
isinstance(a, (int, float)) # 判断a是否是int, float类型

exit # 退出

-----------------------------
### 不懂
regex
import re

re.split('\s+', text)
regex.findall(text)

regex = re.compile('\s+')
regex.split(text)
regex.findall(text)

------------------
a is b # a is not c
a is None

-------------------------
5 / 2   # 2.5
5 // 2  # 2
5 % 2   # 1
1e5 = 100000.0

-------------
'{0:.2f} {1:s} are worth US${2:d}'.format(4.56, 'abc', 1)
# .2f 保留两位小数

---------------------------------------
from datetime import datetime, date, time

---------------------
map
# 将function应用于iterable的每一个元素，结果以列表的形式返回
# 用于series的map方法接受一个函数，或是一个字典，包含着映射关系即可
map(func, iterable, ……) # 函数名, 可以迭代的对象，例如列表，元组，字符串
data.map(func) # 将func应用到data中

-------------------
lambda
# 需要有传入参数
g = lambda x, y, z : (x + y) ** z
print g(1,2,2)

map(lambda x: x * x, [y for y in range(10)])

-------------
a, b = input("请输入一个数字：") # python输入 默认str类型
a, b = map(int, input().split()) # 输入a, b 转换为int 

str = input() # 'a b cde'
strlist = input().split(' ')	 # ['a', 'b', 'cde']
str1, str2 = input().split(' ')  # 输入两个字符串，使用空格分隔输入

intnum = int(input())
intlist = list(map(int, input().split(' ')))  

print(a, end = ' ')
-----------------------------------
# print的返回值 == None
# 对于所有没有return的函数，自动加上 return None
