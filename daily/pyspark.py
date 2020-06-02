***Dataframe***
df = hc.sql('')
df.show()
df.columns
df.printSchema()
df.collect()
df.toPandas()
df.count()

-------------------
# shift + tab 反向缩进
# 选中 tab 多行缩进
 
-------------------
sc = SparkContext(conf=)
hc = HiveContext(sc)
df.registerTempTable('t')
hc.sql('''drop table if exists dbname''')
hc.sql('''create table dbname as select * from t''')

---------------------
#建立新列
df.withColumn('new_col',fn.monotonically_increasing_id())

df.where('col_name == 2')

df.select('col_name1', 'col_name2')

df.select([c for c in df.columns if c != 'col_name'])

df.drop('col_name')
df.drop(df.col_name)

cond = [df.name == df3.name, df.age == df3.age] 
df.join(df3, cond, 'outer')

-------------------------
import pyspark.sql.functions as fn
pd.agg(fn.count('col_name').alias('new_name'),
	   fn.countDistinct('col_name').alias('distin_name')).show()

-----------------------------
df.groupby('col_name').count().show()
df.agg(min(col('col_name'))).show()


-------------------
*时间( )

#提取年月日xxxx-xx-xx
df.select(to_date(df.col_time).alias(col_name))

df.withColumn('new_col',to_date(df.col_time)) # 返回添加new_col后的全量df

df.select(
	year('col_time').alias('new_name'),
	month('col_time').alias('new_name'),
	day('col_time').alias('new_name')
)

-----------------------

df.agg(*[
	(1 - (fn.count(c)/fn.count('*'))).alias(c+'_missing')
	for c in df.columns
])#缺失值比例

------------------------
df.filter((df['col_name1'] < 200) & (df['col_name2'] > 200))
df.filter("col_name1 < 500").select(['col_name1','col_name2'])
