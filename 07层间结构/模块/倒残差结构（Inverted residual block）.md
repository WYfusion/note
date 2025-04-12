在**MobileNet**中提出

与残差结构很相似，但是主分支是**先升维，再卷积，后降维**

MobileNet的倒残差结构采用DW卷积和ReLu6激活函数，即$y=\mathrm{ReLU}6(x)=\min(\max(x,0),6)$，也即仅在$[0,6]$之间有$y=x$

<img src="../../assets/image-20241114185047599.png" alt="image-20241114185047599" style="zoom: 33%;display: block; margin: 0 auto;" />