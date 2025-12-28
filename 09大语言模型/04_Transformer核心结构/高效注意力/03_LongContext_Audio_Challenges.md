# 长序列挑战：语音特有的上下文问题

相比于文本，语音信号的信息密度极低（Low Information Density），导致序列长度极长。这给 Transformer 的应用带来了独特挑战。

## 1. 序列长度爆炸

*   **文本**: 1 个 Token $\approx$ 0.75 个单词。1000 个 Token 可以写一篇短文。
*   **语音**:
    *   采样率 16kHz，1 秒有 16,000 个采样点。
    *   STFT 帧移 10ms，1 秒有 100 帧。
    *   1 分钟 = 6,000 帧。
    *   1 小时 = 360,000 帧。

对于标准 Transformer ($O(L^2)$)，处理 1 小时的音频在计算上是不可行的。

## 2. 解决方案一：下采样 (Subsampling)

最直接的方法是减少 $L$。
*   **卷积下采样**: 在 Transformer 之前，使用 2 层 Stride=2 的 CNN，将帧率降低 4 倍（从 10ms/帧 变为 40ms/帧）。Whisper 就采用了这种策略。
*   **金字塔结构 (Pyramid)**: 像 Conformer 或 Audio Pyramid Transformer 那样，随着层数加深，逐渐合并相邻的时间步。

## 3. 解决方案二：局部注意力 (Window Attention)

语音具有很强的**局部性**。识别当前的音素，通常只需要前后几百毫秒的信息。
*   **Sliding Window**: 每个 Token 只关注自己窗口内的 $w$ 个 Token。
*   **Longformer / BigBird**: 结合局部窗口和稀疏的全局 Token（Global Tokens）。

## 4. 解决方案三：分块处理 (Chunk-wise Processing)

对于无限长的流式音频，必须分块。
*   **Chunk-wise Attention**: 将音频切分为固定长度的 Chunk（如 10秒）。
*   **Context Carry-over**: 将上一个 Chunk 的 KV Cache 传递给下一个 Chunk，以保持跨 Chunk 的上下文信息（Transformer-XL 的思想）。

## 5. 总结

在设计语音大模型时，必须根据任务需求选择合适的策略：
*   **短语音指令**: 标准 Attention 即可。
*   **长会议转录**: 需要 FlashAttention + 下采样 + 滑动窗口。
*   **实时流式对话**: 需要 Block-wise 处理或 Linear Attention。
