# 数据来源和处理方式

# 数据预处理

行业未分类的处理


# TODO 回归及解释

# TODO 特征工程 - 变量筛选

使用每个变量的平均值

分类变量 onehot cardinal-encoding

排名靠前特征暴力交叉

每个特征工程前后效果对比 + 全部特征工程前后对比

# 实证

相关性分析 方差分析

shap 对个股的分析

# 回归分析基本流程

# 异常检测

异常检测算法

# 集成模型

# 分布式计算

# 时间序列方法

# TODO

check_fairness 不同行业准确率

模型修正理论补充

使用雷达图画几个重要特征看规律 Shapiro-Wilk计算特征重要性 https://zhuanlan.zhihu.com/p/78351708

before after calibration 图形上看倒退了


KS取值范围	模型效果
KS<0.2	无区分能力
0.2≤KS<0.3	模型具有一定区分能力，勉强可以接受
0.3≤KS<0.5	模型有较强区分能力
0.5≤KS<0.75	模型具有很强区分能力
0.75≤KS	模型可能有异常（效果太好，以至于可能有问题）

lift 曲线 https://blog.csdn.net/some_apples/article/details/104166842

manifold 貌似没有用 后期可以探索一下

YellowBrick 的文档可以看到各种图的解释

决策树的可视化 pot tree learning curve

shap reason plot  x roe y pe 明显可解释

最后出效应结果时将股价也进行展示

baostock 官网有对指标含义的介绍，可合并进来

合并报表与母公司报表 归母净利润与净利润

中国石油异常值问题

分布式计算

研究缺陷 未加入市场宏观指标

股价 和 成长性组合：净利润 主营业务收入 净资产 总资产 利用成长性组合预测股价涨跌

结果图可以 use_train_data 比如 AUC

deepcheck

calibration 是不是仅针对 sotonic or logistic regression

learning validation curve 不同模型出不同的图 循环
