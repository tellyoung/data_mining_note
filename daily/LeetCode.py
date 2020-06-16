LeetCode

not None True False

True + True = 2
min(a, b), max(a, b), abs(a)
bin(a ^ b) # 前缀'0b'开头 二进制
int('1010', 2) # 转换二进制str => 十进制int

'{0:b}'.format(int(str1, 2) + int(str2, 2)) #二进制加法

range(5, -1, -1) # 5, 4, 3, 2, 1, 0
9 // 4 = 2 # int类型
6 / 2 = 3.0 # 浮点数
最小值：Integer.MIN_VALUE= -2147483648 （-2的31次方）
最大值：Integer.MAX_VALUE= 2147483647  （2的31次方-1）
int('-' + '231')
Out[14]: -231


(((((((((((((((((((((((((((((())))))))))))))))))))))))))))))
----------------------------------------------------
n&1   # 如果是偶数，n&1返回0；否则返回1，为奇数
n>>1  # 向下整除2
如果a，b是数值变量,&， |表示位运算， and，or则依据是否非0来决定输出
and中含0，返回0； 均为非0时，返回后一个值
or中， 至少有一个非0时，返回第一个非0
如何a, b是逻辑变量， 则两类的用法基本一致
-------------------------------------------------------

pow(x,n) # x的n次方
return abs(self.depth(root.left) - self.depth(root.right)) <= 1 and \   #写不下加\
    self.isBalanced(root.left) and self.isBalanced(root.right)
---------------
copy.deepcopy(head) # 深拷贝复杂链表

--------
l = []
[0] * 10
l.append(x)   # l.append([])
l.extend([2,3,4])  # 
l.pop(0)      # 弹出0号位置 None也是一个元素不为空
l.extend([a, b, c])  # 添加a, b, c加入l
l.remove('a')
'a' in l
row = len(l)  # 二维数组 col = len(l[0])
l.sort(key=)  # 默认升序 / key=len，
min(l)
sorted(l)
[str(i) for i in l]
l[0:3]         # 0,1,2 不包括位置3 / [::2] / [::-1]倒序
l[::-1]  #倒序
l.index(values) # 返回在数组中的位置
for index, value in enumerate(some_list): # 枚举
    mapping[value] = index # mapping = {}

for i, (a, b) in enumerate(zip(l1, l2)):
    print('{0}: {1}, {2}'.format(i, a, b))

zipped = zip(l1, l2)
zip(*zipped)

list(reversed(range(10))) # 倒序9-0

dp = [None]*(n+1)
dp[0] = 0
dp[1] = 1
------------------------
from queue import Queue

q = Queue()
q.put(x)     # 添加x进入队列 only one？
q.get()		 # 获取队列弹出值 队首移出队列
q.empty() 	 # 判断是否为空 不为空return False
q.qsize()    # return the size of queue


d = collections.deque([])
d.append('a') # 在最右边添加一个元素，此时 d=deque('a')
d.appendleft('b') # 在最左边添加一个元素，此时 d=deque(['b', 'a'])
d.extend(['c','d']) # 在最右边添加所有元素，此时 d=deque(['b', 'a', 'c', 'd'])
d.extendleft(['e','f']) # 在最左边添加所有元素，此时 d=deque(['f', 'e', 'b', 'a', 'c', 'd'])
d.pop() # 将最右边的元素取出，返回 'd'，此时 d=deque(['f', 'e', 'b', 'a', 'c'])
d.popleft() # 将最左边的元素取出，返回 'f'，此时 d=deque(['e', 'b', 'a', 'c'])
d.rotate(-2) # 向左旋转两个位置（正数则向右旋转），此时 d=deque(['a', 'c', 'e', 'b'])
d.count('a') # 队列中'a'的个数，返回 1
d.remove('c') # 从队列中将'c'删除，此时 d=deque(['a', 'e', 'b'])
d.reverse() # 将队列倒序，此时 d=deque(['b', 'e', 'a'])


---------------------
s = "string"
str_list = ['aa', 'ab', 'ac']
s.isalnum() # 全由字母或数字组成 True / s[3].isalnum()
s.isdigit() # 数字
s.isalpha()

s.upper()   # 转换为大写
s.title() #首字母大写
'a' + 'b' + 'c' #'abc'
'i' in s
s.index('r') # s.find('r')无则-1 / 均返回第一个'r'在s中的位置
"".join(["a", "b", "c"]) # "abc" # "/".join(["a", "b", "c"]) # "a/b/c"

# 把列表中分开的字符串合并
"a/b/c".split("/") # 以"/"分离 ['a', 'b', 'c']
" a b c".strip() #  去头空格"a b c"
s.count('s') # 's'出现次数
s.replace('s', 'S') # 'String' # s.replace('s', '') # 'tring'
'abc' * 2 # 'abcabc'

str = "00000003210Runoob01230000000"; 
print str.strip( '0' );  # 去除首尾字符 0
 
s.replace(' ', '%20') # 字符串中的空格用%20替换
 
str2 = "   Runoob      ";   # 去除首尾空格
print str2.strip();

--------------------
set
s = set(('str', 'str1', 'str2')) # s = {'str', 'str1', 'str2'}
s = set(list)
s = {(i, j)} # s = set([1, 2, 3]) 等价 s = {1, 2, 3}
s.add((di, dj))

s.discard(value) # 抛弃value, 如果元素不存在，不会发生错误
s.pop()
s.clear()
s1 = {(i, j) for i in range(row) for j in range(col) if grid[i][j] == 2}
s2 = {(i + di, j + dj) for i, j in s1 for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (i + di, j + dj) in s3}
s1 -= s2
s1.union(s2) # 返回并集 s1 | s2
s2.intersection(s2) # 返回交集 s1 & s2
s1.issubset(s2) # s1是否是s2的子集

import collections
collections.Counter([2,2,1,1,1,2,2]) # Counter({2: 4, 1: 3}) 字典类型

-------------
dict
d = {}
'key' in d # 'key' not in d
del d['key']
d.keys()
d.values()
d.update({'key':values, })
d = {index: value for index, value in enumerate(s)} # s : string
for i in d: # i指代d.keys

collections.Counter('dsf')
Counter({'d': 1, 's': 1, 'f': 1})

d.update(c=3, d=4)
d[key] = value
print(d)
# {'a': 1, 'c': 3, 'b': 2, 'd': 4}

-----------
heapq # 最小堆
import heapq
heap = [1,2,3,5,1,5,8,9,6]
heapq.heappush(heap, item)  # heap为定义堆，item增加的元素

heapq.heapify(heap) # 将列表转换为堆
x = heapq.heappop(heap) # 弹出堆顶最小值，重建堆
x = heapq.heapreplace(heap, item) #  弹出最小元素值，替换新的元素值，重建堆
heapq.merge(heap1, heap2) # 合并两个堆
for i in heapq.merge(heap1, heap2):
    print(i, end=" ")
heapq.nlargest(n, heap)  # n个最大元素 
heapq.nsmallest(n, heap)  # n个最小元素



记录路径时若直接执行res.append(path) ，则是将 path 列表对象 加入了 res ；
后续 path 对象改变时， res 中的 path 对象 也会随之改变（因此肯定是不对的，
本来存的是正确的路径 path ，后面又 append 又 pop 的，就破坏了这个正确路径）。
list(path) 相当于新建并复制了一个 path 列表，因此不会受到 path 变化的影响。
------------------
'''
使用颜色标记节点的状态，新节点为白色，已访问的节点为灰色。
如果遇到的节点为白色，则将其标记为灰色，然后将其右子节点、自身、左子节点依次入栈。
如果遇到的节点为灰色，则将节点的值输出。
'''
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res

----
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        result = []
        def inorder(root):
            if not root:
                return None
            inorder(root.left)
            result.append(root.val)
            inorder(root.right)
        inorder(root)
        return result

---------
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        levelx = []
        if not root:
            return None
        def fun_level(node, level):
            if len(levelx) == level:
                levelx.append([])
            levelx[level].append(node.val)
            if node.left:
                fun_level(node.left, level + 1)
            if node.right:
                fun_level(node.right, level + 1)
        fun_level(root, 0)
        return levelx


class Solution:
    def spiralOrder(self, matrix:[[int]]) -> [int]:
        if not matrix: return []
        l, r, t, b, res = 0, len(matrix[0]) - 1, 0, len(matrix) - 1, []
        while True:
            for i in range(l, r + 1): res.append(matrix[t][i]) # left to right
            t += 1
            if t > b: break
            for i in range(t, b + 1): res.append(matrix[i][r]) # top to bottom
            r -= 1
            if l > r: break
            for i in range(r, l - 1, -1): res.append(matrix[b][i]) # right to left
            b -= 1
            if t > b: break
            for i in range(b, t - 1, -1): res.append(matrix[i][l]) # bottom to top
            l += 1
            if l > r: break
        return res
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        left, right, top, bot = 0, len(matrix[0]), 0, len(matrix)
        while left <= right and top <= bot:
            for i in range(left, right):
                res.append(matrix[top][i])

            for i in range(top + 1, bot):
                res.append(matrix[i][right - 1])

            for i in range(right - 2, left, -1):
                res.append(matrix[bot - 1][i])

            for i in range(bot - 1, top, -1):
                res.append(matrix[i][left])
            
            left += 1
            right -= 1
            top -= 1
            bot += 1

        return res


