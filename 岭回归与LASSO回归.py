'''
本章所用模块，函数及方法：
        sklearn：
                    Rideg：用于设定岭回归模型的”类“
                    RidgeCV：用于设定岭回归交叉验证的”类“
                    Lasso：用于设定LASSO回归模型的”类“
                    LassoCV：勇于设定Lasso回归交叉验证的”类“
                    fit：基于”类“的模型拟合”方法”
                    alpha_：用千返回岭回归与LASSO回归的自变量系数
                    predict：基千模型的预测 ”方法”
                    mean_squared_error：计算均方误差MSE的函数，如需计算RMSE, 还需对其开根号
        statsmodels：
                    add_constant：用千给数组添加常数列1的函数
                    ols：用千设定多元线性回归模型的 “ 类 ”
'''

# 导入第三方模块
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt

# 读取糖尿病数据集
diabetes = pd.read_excel(r'E:\python\data codes\codes\第8章 岭回归与LASSO回归模型\diabetes.xlsx')
# 构造自变量（剔除患者性别、年龄和因变量）
#predictors表示取diabetes的第三列至倒数第二列的列标，即为column，
# 2到-1表示从左右两头取，同样可以用[2：10]表示，但此处取不到10，是左闭右开
predictors = diabetes.columns[2:-1]
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(diabetes[predictors], diabetes['Y'],
                                                                    test_size=0.2, random_state=1234)
#此处train_test_split加入了两组源数据是因为要赋值给x与y



# 构造不同的Lambda值（对数刻度，下面为200个数字）
Lambdas = np.logspace(-5, 2, 200)
# 构造空列表，用于存储模型的偏回归系数
ridge_coefficients = []
# 循环迭代不同的Lambda值(将之标准化，迭代lambda，计算出每个lambda对应的各变量回归系数)
for Lambda in Lambdas:
    ridge = Ridge(alpha=Lambda, normalize=True)
    ridge.fit(X_train, y_train)
    ridge_coefficients.append(ridge.coef_)

# 绘制Lambda与回归系数的关系
# 中文乱码和坐标轴负号的处理（一般遇到此都可用）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置绘图风格
plt.style.use('ggplot')
plt.plot(Lambdas, ridge_coefficients)
# 对x轴作对数变换
plt.xscale('log')
# 设置折线图x轴和y轴标签
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
# 图形显示
plt.show()



# 岭回归模型的交叉验证
# 设置交叉验证的参数，对于每一个Lambda值，都执行10重交叉验证
ridge_cv = RidgeCV(alphas=Lambdas, normalize=True, scoring='neg_mean_squared_error', cv=10)
# 模型拟合
ridge_cv.fit(X_train, y_train)
# 返回最佳的lambda值
ridge_best_Lambda = ridge_cv.alpha_
print(ridge_best_Lambda)
#%%
# 导入第三方包中的函数
from sklearn.metrics import mean_squared_error

# 基于最佳的Lambda值建模
ridge = Ridge(alpha=ridge_best_Lambda, normalize=True)
ridge.fit(X_train, y_train)
# 返回岭回归系数
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[ridge.intercept_] + ridge.coef_.tolist()))
# 预测
ridge_predict = ridge.predict(X_test)
# 预测效果验证
RMSE = np.sqrt(mean_squared_error(y_test, ridge_predict))
RMSE
#RMSE表示均方误差，其值越小，表示预测效果越显著。


#%%
# 导入第三方模块中的函数
from sklearn.linear_model import Lasso, LassoCV

# 构造空列表，用于存储模型的偏回归系数
#iter表示迭代器，max_iter表示最大迭代次数
lasso_coefficients = []
for Lambda in Lambdas:
    lasso = Lasso(alpha=Lambda, normalize=True, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_coefficients.append(lasso.coef_)

# 绘制Lambda与回归系数的关系
plt.plot(Lambdas, lasso_coefficients)
# 对x轴作对数变换
plt.xscale('log')
# 设置折线图x轴和y轴标签
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
# 显示图形
plt.show()
#%%
# LASSO回归模型的交叉验证
lasso_cv = LassoCV(alphas=Lambdas, normalize=True, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)
# 输出最佳的lambda值
lasso_best_alpha = lasso_cv.alpha_
print(lasso_best_alpha)

# 基于最佳的lambda值建模
lasso = Lasso(alpha=lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X_train, y_train)
# 返回LASSO回归的系数
print(pd.Series(index=['Intercept'] + X_train.columns.tolist(), data=[lasso.intercept_] + lasso.coef_.tolist()))
#coef_表示实例函数系数，tolist表示将数组或矩阵转换为列表

# 预测
lasso_predict = lasso.predict(X_test)
# 预测效果验证
print(RMSE = np.sqrt(mean_squared_error(y_test, lasso_predict)))

#%%
# 导入第三方模块
from statsmodels import api as sms
# 为自变量X添加常数列1，用于拟合截距项
X_train2 = sms.add_constant(X_train)
X_test2 = sms.add_constant(X_test)
x=pd.concat([X_train2,y_train],axis=1)
# 构建多元线性回归模型
linear = sms.formula.ols('Y~BMI+BP+S1+S2+S3+S4+S5+S6',data=x).fit()
# 返回线性回归模型的系数
print(linear.params)

# 模型的预测
linear_predict = linear.predict(X_test2)
# 预测效果验证
print(RMSE = np.sqrt(mean_squared_error(y_test,linear_predict)))
