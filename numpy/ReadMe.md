## <center>Numpy</center>

### 创建numpy.array的方法

- 使用list创建
- np.zeros
  - np.zeros(10) # 创建一个包含10个0的数组
  - np.zeros(10,dtype=int) # 指定数组内的类型
  - np.zeros(shape=(3,5)) # 创建3x5的矩阵
- np.ones同上
- np.full(shape=(3,5),file_value=666) # 表示创建3x5的矩阵，里面的数据都被666填充

### arange

np.arange(0,20,2)同python中[i for i in range(0,20,10)]作用相同。与python不同的情况是python中的range方法的步长不可以为浮点数，但是numpy的arange方法步长可以为浮点数

### linspace

linspace其实是lineaspace的缩写。大体用法与**arange**一致。np.linspace(0,20,10)的作用是在[0,20]取出距离相同的10个点（包含0和20）。

### random

- np.random.randint(0,10) # 从0到10中随机生成一个数字
- np.random.randint(0,10,10) # 生成一个长度为10的list，其中的每个元素是0到10的随机数
- 在运行randint方法之前，可以先设置随机数种子，便于调试。np.random.seed
- Np.random.random() # 默认随机生成的随机数
- np.random.normal(10,100,size) # 生成size形状的符合正态分布的随机数

### Numpy.array的基本操作

- 基本属性

  - x.ndim  # 返回的是一个数字，表示有多少个维度
  - x.shape # 返回的是一个tuple
  - x.size # 共有几个元素

- 数据访问

  - 使用python的方式
  - 使用形如X[2,2]或X[(2,2)]这种方式访问多维数组
  - 类似python中的切片，在二维数组中X[:2,:3]访问前两行的前三列
  - 在numpy中改变子矩阵的元素会同时将原矩阵的内容改变
  - 但是调用reshape方法不会改变原数组

- 合并与分割操作

  - np.concatenate([x,y],axis=)

     将列表中的元素进行合并，假如原来是两个1x3的array，该操作之后就变为1x6的操作。这个操作可以理解为是样本的拼接，若是样本特征的拼接，则将axis设置为1，表示使用列拼接，若将axis=0则表示不同的样本直接拼接，也就是以行为轴进行拼接。

  - np.vstack

    v表示vertical，表示在垂直方向上进行堆叠。因为concatenate方法只能处理维度相同的元素。若有形如[[1,2,3],[4,5,6]]和[7,8,9]进行合并的时候必须要将[7,8,9]进行reshape才可以，但是vstack可以直接进行拼接。

  - np.hstack

    与vstack相对应

  - np.split

    共有三个参数。第一个是表示被分割的元素，第二个参数是一个list表示分割点。如np.split(x,[2,3])表示将x分割为三段.第三个参数是分割轴

  - np.hsplit

    在水平方向上分割

  - np.vsplit

    在垂直方向分割

- 具体运算

  - Universal function

    在numpy中存在universal function的说法，即加入X是一个矩阵，则X-1表示X中的所有元素都减1，其他操作同理。

  - 两个矩阵的运算

    在numpy中对于所有运算符的定义都是对应元素进行相应的运算。如X*Y表示的是X对应的元素与Y对应的元素相乘。

  - 向量和矩阵的运算

    向量和矩阵的运算numpy中会将向量与矩阵的每一个元素进行相应的运算

    - np.tile

      np.tile是指堆叠的方法，有两个参数，第一个参数表示要堆叠的对象，第二个参数是一个元祖，表示在行方向堆叠几次，在列方向堆叠几次。如v=[1,2]，则np.tile(v,(2,1))的结果为[[1,2],[1,2]]

  - np.linalg.inv

    矩阵的逆

  - np.linalg.pinv

    因为逆矩阵只有方阵有，当不是方阵的时候想要求一个伪逆矩阵的话使用pinv的方法。

  - np.mean() # 均值

  - np.median() # 中位数

  - Np.percentile() # 百分位点

### Fancy Indexing

### matplotlib

- 指定x轴y轴的范围
  - plt.xlim(-5,15)
  - 可以通过plt.axis([]) 列表中可以传入四个数组，前两个数字表示x后两个表示y
- 线段样式 linestyle
- 标明x轴和y轴代表什么
  - plt.xlabel("") 参数是字符串
- plt.plot(x,siny,label="six(x)") 其中x和y分别表示点的列表，而label表示这条线的说明。还要在show()方法之前加上一个plt.legend()的方法，然后即可在图中显示。
- 图的标题 plt.title("")

