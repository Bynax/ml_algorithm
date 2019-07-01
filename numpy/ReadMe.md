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