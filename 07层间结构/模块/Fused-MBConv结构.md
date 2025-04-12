DW卷积虽然理论计算量小，但是无法充分利用现有的一些加速器，实际上也没有那么快。

<img src="../../assets/image-20241119215204547.png" alt="image-20241119215204547" style="zoom:50%;display: block; margin: 0 auto;" />

**MBConv**结构和**Fused-MBConv**结构对比，将升维的(1×1)卷积和DW卷积替换为了(3×3)卷积。

实际代码中可能并没有SE模块(效果可能不会更好)

实际代码所使用的Fused-MBConv结构：

<img src="../../assets/image-20241119225931388.png" alt="image-20241119225931388" style="zoom: 50%;display: block; margin: 0 auto;" />

注意是否进行了通道扩展，即隐藏层通道扩展因子是否为1
