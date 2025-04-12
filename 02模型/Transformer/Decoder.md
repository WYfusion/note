
![[Pasted image 20250315170430.png|800]]
注意这里使用的是$Masked\ Multi-Head\ Attention$而不是$Multi-Head\ Attention$，区别如下：
![[Pasted image 20250315171643.png|800]]
这里$Masked\ Multi-Head\ Attention$*无法*看到当前时刻**当前位置之后**的所有输入位置信息，而$Multi-Head\ Attention$可以根据全部的输入信息推断输出。$Decoder$是依次产生的，不像$Encoder$是一次性全部生成所有的输出。

常遇到的情况是$Decoder$必须自己决定输出的seq的长度，当前一些情况下也可以指定输出长度。