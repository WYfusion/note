# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE 是一种广泛用于**自动文摘**（Text Summarization）和**机器翻译**领域的自动评价指标。与 BLEU 侧重于**精度**（Precision）不同，ROUGE 更加侧重于**召回率**（Recall），即衡量模型生成的摘要在多大程度上覆盖了参考摘要的内容。

---

## **ROUGE 的核心思想**
ROUGE 通过计算**模型生成的摘要**（System Summary）与**人工撰写的参考摘要**（Reference Summary）之间的 n-gram 重叠程度来评价摘要的质量。它主要关注参考摘要中有多少 n-gram 出现在了生成摘要中。

---

## **常见的 ROUGE 指标**

### 1. ROUGE-N (n-gram Co-occurrence Statistics)
基于 n-gram 共现统计的指标，最常用的是 **ROUGE-1** 和 **ROUGE-2**。

#### **计算公式**
$$
\text{ROUGE-N} = \frac{\sum_\limits{S \in \{\text{Reference Summaries}\}} \sum_\limits{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_\limits{S \in \{\text{Reference Summaries}\}} \sum_\limits{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
$$

*   **分子**：生成摘要与参考摘要共有的 n-gram 个数。
*   **分母**：**参考摘要**中 n-gram 的总个数（体现了召回率的思想）。

#### **解释**
*   **ROUGE-1**：衡量**单个词**（unigram）的重叠，主要反映**信息量**的覆盖程度。
*   **ROUGE-2**：衡量**双词短语**（bigram）的重叠，主要反映**流畅度**和**语义连贯性**。

---

### 2. ROUGE-L (Longest Common Subsequence)
基于**最长公共子序列**（LCS）的指标。LCS 不需要 n-gram 连续匹配，只需要字符（或词）在序列中保持相对顺序即可。

#### **核心优势**
*   自动捕捉最长的共现 n-gram，无需预定义 n。
*   能够更好地反映句子级结构相似性。

#### **计算公式**
假设参考摘要 $X$ 长度为 $m$，生成摘要 $Y$ 长度为 $n$。
1.  **召回率 ($R_{lcs}$)**:
    $$ R_{lcs} = \frac{LCS(X, Y)}{m} $$
2.  **精度 ($P_{lcs}$)**:
    $$ P_{lcs} = \frac{LCS(X, Y)}{n} $$
3.  **F-measure ($F_{lcs}$)**:
    $$ F_{lcs} = \frac{(1 + \beta^2) R_{lcs} P_{lcs}}{R_{lcs} + \beta^2 P_{lcs}} $$
    *   $LCS(X, Y)$ 是 $X$ 和 $Y$ 的最长公共子序列长度。
    *   $\beta$ 用于控制精度和召回率的权重（通常设为很大，强调召回率）。

---

### 3. ROUGE-W (Weighted LCS)
加权最长公共子序列。
*   **问题**：ROUGE-L 对连续匹配和不连续匹配一视同仁。例如 `A B C` 和 `A ... B ... C` 的 LCS 长度相同，但前者显然更好。
*   **改进**：ROUGE-W 对**连续**的匹配给予更高的权重。

---

## **ROUGE vs BLEU**

| 特性 | BLEU | ROUGE |
| :--- | :--- | :--- |
| **侧重点** | **精度 (Precision)** | **召回率 (Recall)** |
| **分母** | 生成文本的 n-gram 总数 | 参考文本的 n-gram 总数 |
| **主要应用** | 机器翻译 (Machine Translation) | 文本摘要 (Text Summarization) |
| **直观理解** | “生成的句子里有多少是对的？” | “参考答案里的内容，你覆盖了多少？” |

---

## **局限性**
1.  **依赖字面匹配**：无法识别同义词或释义（Paraphrasing）。例如“聪明”和“智慧”会被视为不匹配。
2.  **忽视连贯性**：ROUGE-N 仅关注局部 n-gram，可能无法很好地评价长文本的逻辑连贯性。
3.  **参考摘要的主观性**：人工摘要本身具有主观性，不同的参考摘要可能导致分数波动。
