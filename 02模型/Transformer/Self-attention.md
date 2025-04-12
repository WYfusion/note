query、key、value
[台大李宏毅自注意力机制](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=11)
[对应PDF](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa29uS0g1X09qRmlFYXdidUNTVnFwZU4xQmp3UXxBQ3Jtc0tuT19rMThKSXJQekp6VGNDaktCMXA3eHZNX0VVa3JEaWd5VXl3aFpiZVVWR3IxYmU5V1JDRUVLRmNhTzVsSXM1Q0E0dkZ0ZjRPZXhYLTFYdW05UjhZRVZsck92a0I3bmJYWXU1NkR4Mi1yRkRGcDBNRQ&q=https%3A%2F%2Fspeech.ee.ntu.edu.tw%2F%7Ehylee%2Fml%2Fml2021-course-data%2Fself_v7.pdf&v=gmsMY5kc-zw)
### 模型输入输出个数
#### 每一个Vector对应一个label
![[Pasted image 20250314201007.png|400]]
#### 整个输入对应一个输出label
![[Pasted image 20250314201021.png|400]]
#### 模型自己决定label的数量
![[Pasted image 20250314201038.png|400]]
对于语音方面的深度学习过程，一般情况下输入和输出的序列长度是可变的
## 注意力过程示例
假设为输入4个、输出也是4个的情况一。
![[Pasted image 20250314201253.png|500]]
主要使用[[Dot-product]]作为实现计算不同vector之间相关性的方法。
对于输入的$a_1$进行分析：
1. 关联度计算
也可以计算自身的关联性。关联性使用$k$来示意。
![[Pasted image 20250314200100.png|600]]
2. 激活函数处理
获得注意力得分后，再使用激活函数处理，常规使用的是$Soft-max$，也可以换用其他的来试试。
![[Pasted image 20250314200618.png|600]]
3. 提取信息
再添加$v$矩阵，提取注意力得分信息
![[Pasted image 20250314201607.png|600]]

同理对于$a_2$也有相同的上述操作，输出得到$b_2$。直至得到目标输出$b$个。
![[Pasted image 20250314201918.png|600]]
## 矩阵表示
#### 总体矩阵框架
![[Pasted image 20250314210038.png|600]]
### 由权重矩阵计算Q、K、V
这里的矩阵$W^q、W^k、W^v$在同一时刻、同一个Self-attention中是相同的，而且是在训练的过程中需要学习的。
![[Pasted image 20250314204512.png|700]]
### 由Q、K计算注意力得分
$$A=K^TQ$$
得到注意力得分后，按列进行激活函数操作得到$A^`$。
![[Pasted image 20250314205248.png|700]]

### 由V计算输出特征
$$
O=VA^`
$$
![[Pasted image 20250314205739.png|700]]

### CNN 与 Self-Attention 之间的区别
###### 感受野
- CNN的感受野是人为指定的
- Self-Attention的感受野是模型自己学的

###### 过拟合
由于Self-Attention的感受动态范围比CNN大，所以说一般需要较多的数据来训练

##### Conformer
可以使用CNN与Self-Attention的结合