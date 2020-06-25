LeetCode

---------------
copy.deepcopy(head) # 深拷贝复杂链表

--------
l.sort(key=)  # 默认升序 / key=len，
sorted(l)

[str(i) for i in l]

for index, value in enumerate(some_list): # 枚举
    mapping[value] = index # mapping = {}

for i, (a, b) in enumerate(zip(l1, l2)):
    print('{0}: {1}, {2}'.format(i, a, b))

zipped = zip(l1, l2)
zip(*zipped)

list(reversed(range(10))) # 倒序9-0


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
set
s1 = {(i, j) for i in range(row) for j in range(col) if grid[i][j] == 2}
s2 = {(i + di, j + dj) for i, j in s1 for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)] if (i + di, j + dj) in s3}

import collections
collections.Counter([2,2,1,1,1,2,2]) # Counter({2: 4, 1: 3}) 字典类型
-------------
dict

d = {index: value for index, value in enumerate(s)} # s : string

collections.Counter('dsf')
Counter({'d': 1, 's': 1, 'f': 1})

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
