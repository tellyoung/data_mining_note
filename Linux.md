[toc]
# linux

## 设备信息
```shell
lscpu       # cpu信息
man lscpu   # 指令解释: man 指令

df -h       # 查看硬盘使用情况
```

## 基本命令
```shell
pwd   # 当前路径
ls    # ll详细信息
cd .. # 回到上一级
'     # 进入quote，再按 ' 退出

mkdir folder   # 新建文件夹
sz filename    # 下载filename到本地

# 删除
rm filename   # 删除文件
rm -f folder  # 删除文件夹

# move
mv filename1 ../code/filename2
# filename1 是要转移的文件名
# filename2 转移后的文件名,可与filename1同名

ipython 	# 进入ipython
quit    	# 退出
```

## 压缩
```shell
zip -r new_name_filename need_zip_filename     # 压缩文件need_zip_filename
unzip filename
```

## screen
```shell
screen -S name 			# 新建一个叫name的session(会话)
screen -ls     			# 列出当前所有的session
screen -r name 			# 回到name这个session
screen -d name 			# 远程detach(分离）某个session
screen -d -r name_ 	    # 结束当前session并回到 name_ 这个session
screen -S name -X quit  # kill叫name的session
# 进入某个session后,先按键 ctrl+a, 再按 d 即可挂起(Detached)
```

## vim
```shell
vim filename.py		# 进入vim
i    			    # 进入插入模式

Esc		 	# 回到命令行模式
Esc :wq 	# 保存并退出
	:q! 	# 不保存并强制退出
	:q  	# 不保存文件,退出
```


## top
>VIRT：virtual memory usage 虚拟内存
>RES：resident memory usage 常驻内存
>SHR：shared memory 共享内存
q – 退出 top


## tips
```shell
cat bigname | split -b 2G - smallname. # smallname.aa / ab / ac
# 在windows本地对下载的2G文件重命名为1.zip.001，1.zip.002，1.zip.003 ……
# 打开cmd进入到存放文件目录中输入：
copy /b 1.zip.001+1.zip.002+1.zip.003 1.zip
# 当前目录生成1.zip，为最后的待解压文件
# 文件之间用加号+连接不能有空格, 另一种写法： copy /b 1.* 1.zip
```