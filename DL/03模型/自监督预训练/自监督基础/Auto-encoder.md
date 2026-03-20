2006年就有了Auto-encoder，但是其思想可以划归于Self-supervised Learning。

---
Auto-encoder的框架中包含了Encoder和Decoder，这两者之间可以实现像[[Cycle GAN#^84fa42|Cycle GAN的循环一致性]]的效果。

Encoder的作用是将复杂很多维的特征向量(输入向量)转化为较低维的向量，例如100pix × 100pix的三色图，则有大小为30000维的输入向量。下面用3×3的进行演示，假设实际训练中可能只有两个维度可行。![[Pasted image 20250318131637.png|600]]
# De-noising Auto-encoder
当然也可以在输入中添加一些干扰，再执行Auto-encoder，将Decoder的输出设定为原始未加噪的输入以实现重建。
![[Pasted image 20250318132605.png|600]]