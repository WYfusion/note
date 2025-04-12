Cross Attention是在2016年的ICASS上提出的，核心机制就是在decoder过程需要使用Decoder产生的q与Encoder的k、v做乘积。原标准模型是使用Encoder最后一个Self-attention块的输出作为键（k）和值（v）。
![[Pasted image 20250315215541.png|400]]
![[Pasted image 20250315214850.png|650]]

![[Pasted image 20250315214529.png|1000]]

## 其他的Cross Attention
下面的是使用Encoder中不同的Self-Attention块内的键（k）和值（v）作为Decoder的交叉注意力计算数值 。
![[Pasted image 20250315215652.png]]

