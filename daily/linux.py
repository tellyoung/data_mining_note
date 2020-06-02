
lscpu
man lscpu
free -m
lsblk列出块设备
df -h查看硬盘使用情况
----
ls # ll详细信息

pwd #/root  #/root/yuty
..  #回到上一级 cd ..

" ' " # 进入quote，再入" ' "退出

-------------------
# 压缩file
zip -r new_filename needzip_filename
unzip filename

-------------
rm -f filename # 删除文件
rm -rf folder  # 删除文件夹
sz filename    # 下载filename
mkdir folder   # 新建文件夹

mv filename ../code/filename
--------------------
screen -S name # 新建一个叫name的session(会话)
screen -ls     # 列出当前所有的session
screen -r name # 回到name这个session
screen -d name # 远程detach(分离）某个session
screen -d -r name_ 	   # 结束当前session并回到name_这个session
screen -S name -X quit # kill叫name的session
# 进入session后,先按键 ctrl+a, 再按 d 即可挂起(Detached)

--------------------
ipython
quit #退出

-----------------
vim
vim filename.py # 进入vim
# i 进入插入模式 ，Esc回到命令行模式

Esc :wq 保存并退出
	:q! 不保存并强制退出
	:q  不保存文件,退出

-----------
from common import fetch
#pandas df格式
df = fetch.fetch_from_hive('select * from dbname limit 100')
data_dir = '../data'
for x in table_list:
       df = fetch.fetch_from_hive('select * from {} where dt="2020-03-03"'.format(x))
       df.to_pickle(os.path.join(data_dir, 'df.{}'.format(str(x).split('.')[1])))

----------------
cat bigname | split -b 2G - smallname. # smallname.aa / ab / ac
# 在windows本地对下载的2G文件重命名为1.zip.001，1.zip.002，1.zip.003 ……
# 打开cmd进入到存放文件目录中输入：copy /b 1.zip.001+1.zip.002+1.zip.003 1.zip
# 此时就会在当前目录生成1.zip，为最后的待解压文件
# 文件之间用加号+连接不能有空格, 另一种写法： copy /b 1.* 1.zip

