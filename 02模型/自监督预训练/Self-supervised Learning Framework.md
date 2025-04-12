```dataview
list
from "模型/自监督预训练"
where
    file.name != "Self-supervised Learning Framework"
LIMIT 10
```
用大量的未标记的数据去训练模型，所以必须先找到一些不需要标记数据的任务，例如：填空，预测下个token等等。本身这个任务没有什么用，但是可以用fine-tune微调以实现下游任务的应用。这种自监督学习的技术有Auto-encoder、BERT、GPT等等
![[Pasted image 20250318105908.png|600]]
