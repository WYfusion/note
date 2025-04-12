[台大李宏毅,24min处起]([【機器學習2021】Transformer (上)](https://www.youtube.com/watch?v=n9TlOhRjYoc))
[配套pdf](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbktFNW9aSUxPR2NBYmpnV2ZfZWdnc2FvMXpYd3xBQ3Jtc0tsVl82QkIxVU1OVFA2TGcxNFVwOUs3eVdpZS1McXotS0VHWi1zeUpQTTJFNDhETGsyNjgySzF2a01vXzgzTFBXVXdZUHRLM3djYzF5TVVONTBQTTV5eWRSOVJzNzRwY015N0VWVVZIVWxteVR4QjZFSQ&q=https%3A%2F%2Fspeech.ee.ntu.edu.tw%2F%7Ehylee%2Fml%2Fml2021-course-data%2Fseq2seq_v9.pdf&v=n9TlOhRjYoc)

---


![[Pasted image 20250315151655.png|800]]
---

---
编码器的实质也是一个seq2seq的问题

![[Pasted image 20250315163356.png|800]]

注意，这里的Block**重复了多次**，需要**多个Self-Attention块（Encoder Block）串联**，这种层级堆叠的结构是其核心设计之一。但是Encoder的设计方案还有很多。

![[Pasted image 20250315161056.png|800]]

这里的[[Multi-Head Self-Attention]]内部就是[[Self-attention]]经过特征分头的处理。

注意这里的Norm是[[LN(LayerNormalization)|LayerNormalization]]！不是[[BN(BatchNormalization)|BatchNormalization]]，当然也可以使用其他层归一化方案。

