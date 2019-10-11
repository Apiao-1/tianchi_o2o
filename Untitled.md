# 可以进一步优化的点

1. 对不同时期的特征建模再stacking
2. 对不同正负样本比例的数据集进行建模
3. GPU优化



# Tricks

1.穿越特征：赛题提供的预测集中，包含了同一个用户在整个7月份里的优惠券领取情况，这实际上是一种leakage，比如存在这种情况：某一个用户在7月10日领取了某优惠券，然后在7月12日和7月15日又领取了相同的优惠券，那么7月10日领取的优惠券被核销的可能性就很大了。我们在做特征工程时也注意到了这一点，提取了一些相关的特征。加入这部分特征后，AUC提升了10个百分点，相信大多数队伍都利用了这一leakage，但这些特征在实际业务中是无法获取到的。







# other

对大量离散的特征可以专门建一个基模型把它转化为一个特征

![img](https://img-blog.csdn.net/20161228120444634?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQnJ5YW5fXw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

数据的划分，特征的提取？