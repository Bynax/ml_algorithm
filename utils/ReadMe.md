## <center>机器学习模型常用方法</center>

###训练集测试集划分

### 模型度量指标

- 分类模型度量
  - Accuracy
  - Recall
  - 

- 回归模型度量
  - MSE(mean squared error)

    

  - RMSE(root MSE)

    

  - MAE(mean absolute error)

    

  - R squared

    - R-squared is a statistical measure of how close the data are to the fitted regression line.

    -  It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

    - 根据以上两条定义，我们大体可以知道R-squared其实就是因变量可以被线性模型解释的百分比。一个很直接的理解为

      > R-squared = Explained variation / Total variation

      即：

      R-squared is always between 0 and 100%:

      - 0% indicates that the model explains none of the variability of the response data around its mean.
      - 100% indicates that the model explains all the variability of the response data around its mean.

      

      


### 参考

- [What does r squared tell us?](https://www.youtube.com/watch?v=IMjrEeeDB-Y)
- [R-squared](https://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit)