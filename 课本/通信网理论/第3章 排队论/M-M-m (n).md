解决**M/M/1**系统的**服务质量**与**系统效率**之间的矛盾，必须**压缩排队长度、减小等待时间**。通常可采用两种措施：
**增加窗口数**。增加窗口，减少等待时间。例如，通信网络中，增加信道数量、增加带宽等
**截止排队长度**。也就是采用拒绝型系统。保障被服务的顾客不会长时间等待，拒绝新顾客，**换取系统效率和稳定性**。

### M/M/m (n)排队系统的模型([[排队论基础#^6c847c|混合排队方式]])
![[Pasted image 20250619021307.png|600]]
![[Pasted image 20250619021523.png|600]]
![[Pasted image 20250619074536.png|600]]
[[Pasted image 20250619011023.png|生灭公式3.26]]
1. $P_k = P_m \cdot \rho^{k-m} = \frac{m^m \rho^m}{m!} P_0 \cdot \rho^{k-m} = \frac{m^m}{m!} \rho^k P_0$
2. **计算$P_m$**： 当$k = m$ 时，利用$k \leq m$ 的递推公式：$P_m = \frac{(m\rho)^m}{m!} P_0 = \frac{m^m \rho^m}{m!} P_0$ （此处$(m\rho)^m = m^m \rho^m$，因$\rho = \lambda/(m\mu)$，故$\lambda = m\rho\mu$，代入$k \leq m$ 的通式$\frac{(m\rho)^k}{k!} P_0$ 得$P_m$）。
3. **递推$k \geq m$ 的$P_k$**： 对于$k = m+1, m+2, \ldots, n$，每次状态转移的服务率为$m\mu$，输入率仍为$\lambda = m\rho\mu$，故转移概率为：$\frac{\lambda}{\mu_k} = \frac{m\rho\mu}{m\mu} = \rho$ 因此，$P_{k} = P_{k-1} \cdot \rho$。递推可得：$P_k = P_m \cdot \rho^{k-m} = \frac{m^m \rho^m}{m!} P_0 \cdot \rho^{k-m} = \frac{m^m}{m!} \rho^k P_0$
![[Pasted image 20250619080606.png|600]]
当 $m \leq k \leq n$ 时，$P_k$ 的求和项为 $\sum_\limits{k=m}^{n} \frac{m^m}{m!} \rho^k P_0$。提取公因子 $\frac{m^m}{m!} \rho^m$，剩余部分为 $\sum_\limits{k=m}^{n} \rho^{k-m} = \sum_\limits{i=0}^{n-m} \rho^i$（令 $i = k-m$，则 $i$ 从 $0$ 到 $n-m$，共 $n-m+1$ 项）。
### M/M/m(n)排队系统的主要性能指标
![[Pasted image 20250619011919.png|600]]
![[Pasted image 20250619080808.png|600]]
推导 $L_s$ 的第二个等式时，分两部分处理求和：
1. **第一部分（$0 \leq k \leq m-1$）**： 当 $k=0$ 时项为 0，故从 $k=1$ 开始：$\sum_\limits{k=1}^{m-1} k \frac{(m\rho)^k}{k!} P_0 = \sum_\limits{k=1}^{m-1} \frac{(m\rho)^k}{(k-1)!} P_0$ 化简为 $\sum_\limits{k=0}^{m-1} k \frac{(m\rho)^k}{k!} P_0$（$k=0$ 项自然为 0，不影响求和）。
2. **第二部分（$m \leq k \leq n$）**： 令 $i = k - m$（$i=0,1,\dots,n-m$），则 $k = m + i$，求和式变为：$\sum_\limits{i=0}^{n-m} (m + i) \frac{m^m}{m!} \rho^{m+i} P_0 = \frac{(m\rho)^m}{m!} P_0 \sum_\limits{i=0}^{n-m} (m + i) \rho^i$ 分解为 $m\sum \rho^i + \sum i\rho^i$，利用等比数列求和公式：
    - $\sum_\limits{i=0}^{N} \rho^i = \frac{1 - \rho^{N+1}}{1 - \rho}$（$N = n-m$），
    - $\sum_\limits{i=0}^{N} i\rho^i = \rho \frac{1 - (N+1)\rho^N + N\rho^{N+1}}{(1 - \rho)^2}$。 代入后通分并整理分子：$m(1 - \rho) + \rho - (n+1)\rho^{n-m+1} + n\rho^{n-m+2} = m - (m-1)\rho - (n+1)\rho^{n-m+1} + n\rho^{n-m+2}$ 因此第二部分和为：$\frac{(m\rho)^m}{m!} P_0 \cdot \frac{m - (m-1)\rho - (n+1)\rho^{n-m+1} + n\rho^{n-m+2}}{(1 - \rho)^2}$
3. **合并两部分**： 第一部分的 $\sum_\limits{k=1}^{m-1} \frac{(m\rho)^k}{(k-1)!}$ 与第二部分的化简式相加，即得：$L_s = \left( \sum_\limits{k=1}^{m-1} \frac{(m\rho)^k}{(k-1)!} + \frac{(m\rho)^m}{m!} \cdot \frac{m - (m-1)\rho - (n+1)\rho^{n-m+1} + n\rho^{n-m+2}}{(1 - \rho)^2} \right) P_0$
![[Pasted image 20250619084130.png|600]]
### 即时拒绝系统
![[Pasted image 20250619084343.png|600]]
![[Pasted image 20250619084401.png|600]]
![[Pasted image 20250619084416.png|600]]
![[Pasted image 20250619084530.png|600]]
![[Pasted image 20250619084601.png|600]]

