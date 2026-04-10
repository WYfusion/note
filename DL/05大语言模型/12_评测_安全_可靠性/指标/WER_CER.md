# WER & CER (Word/Character Error Rate)

WER (Word Error Rate) 和 CER (Character Error Rate) 是**语音识别 (ASR)** 和 **语音转文本 (Speech-to-Text)** 任务中最常用的评价指标。它们基于**编辑距离 (Levenshtein Distance)**，衡量模型输出序列与参考序列之间的差异。

---

## **1. WER (Word Error Rate)**

^f61b13

衡量的是**词级别**的错误率。它反映了将识别出的词序列转换为参考词序列所需的最小编辑操作次数。

### **计算公式**
$$
\text{WER} = \frac{S + D + I}{N} \times 100\%
$$

其中：
*   **$S$ (Substitution, 替换)**：被错误识别为其他词的单词数（如 "apple" $\to$ "apply"）。
*   **$D$ (Deletion, 删除)**：在识别结果中漏掉的单词数（如 "eat an apple" $\to$ "eat apple"）。
*   **$I$ (Insertion, 插入)**：在识别结果中多出的单词数（如 "eat apple" $\to$ "eat an apple"）。
*   **$N$ (Number of Words)**：**参考序列**（Reference）中的总单词数。

### **示例推导**
*   **参考文本 (Ref)**: "The quick brown fox" ($N=4$)
*   **识别文本 (Hyp)**: "The **quack** brown **red** fox"

**分析步骤**:
1.  "The" $\to$ "The" (Match)
2.  "quick" $\to$ "**quack**" (Substitution, $S=1$)
3.  "brown" $\to$ "brown" (Match)
4.  (None) $\to$ "**red**" (Insertion, $I=1$)
5.  "fox" $\to$ "fox" (Match)

**计算**:
$$
\text{WER} = \frac{1 (S) + 0 (D) + 1 (I)}{4 (N)} = \frac{2}{4} = 50\%
$$

### **注意事项**
*   **WER 可以超过 100%**：如果插入错误 ($I$) 非常多，分子可能大于分母。
*   **对语序敏感**：WER 严格依赖词序。
*   **文本归一化**：在计算 WER 前，通常需要对文本进行标准化（去除标点、统一大小写、数字转文字等），以避免非识别错误导致的扣分。

---

## **2. CER (Character Error Rate)**

#CER 衡量的是**字符级别**的错误率。计算逻辑与 WER 完全相同，只是操作单元从“词”变成了“字符”。

### **计算公式**
$$
\text{CER} = \frac{S + D + I}{N_{char}} \times 100\%
$$
其中 $N_{char}$ 是参考文本的总字符数。

### **适用场景**
*   **中文/日文等非拉丁语系**：中文通常不以词为空格分隔，且分词存在歧义，因此 CER（字错误率）比 WER 更常用且更准确。
*   **端到端语音识别**：在分析拼写错误或发音相似导致的字符级错误时很有用。

---

## **3. WER vs CER**

| 特性 | WER (词错误率) | CER (字/字符错误率) |
| :--- | :--- | :--- |
| **基本单元** | 单词 (Word) | 字符 (Character) / 汉字 |
| **主要语种** | 英语、法语等拉丁语系 | 中文、日文等 |
| **难度** | 通常 WER > CER (因为一个词错会导致整个词算错) | 通常 CER 较低 |
| **相关性** | 与人类理解程度相关性较强 | 在拼音文字中相关性较弱 |

---

## **4. 变体与相关指标**

### **SER (Sentence Error Rate)**
*   句子错误率：只要句子中有一个词错，整句就算错。
*   要求极其严格，通常用于命令控制等高精度场景。

### **Match Error Rate (MER)**
*   $$ \text{MER} = \frac{S + D + I}{S + D + I + C} $$
*   分母是参考长度和假设长度的并集（Match + Substitution + Deletion + Insertion）。MER 的值永远在 [0, 1] 之间。

### **TF-WER (Time-Frequency WER)**
*   在某些长音频对齐任务中，考虑时间戳的对齐准确性。
