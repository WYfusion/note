![[Pasted image 20250609231935.png|1000]]
### 次梯度与次微分的定义
- 次梯度（Subgradient）：对于凸函数 $f: \mathbb{R}^n \to \mathbb{R}$，向量 $g \in \mathbb{R}^n$ 是 f 在点 x 处的次梯度，当且仅当对任意 $y \in \mathbb{R}^n$，有：$f(y) \geq f(x) + g^\top (y - x)$ 几何意义：g 定义了 f 在 x 处的**支撑超平面**（仿射下界），该超平面在 x 处与 f 的图像接触，且始终位于 f 图像下方。
- 次微分（Subdifferential）：所有次梯度 g 的集合，记为 $\partial f(x)$。
    - **可导点**：若 f 在 x 处可导，则 $\partial f(x) = \{\nabla f(x)\}$（唯一支撑超平面，即切线）。
    - **不可导点**：存在多个支撑超平面，次微分是这些超平面斜率（或梯度向量）的集合（如绝对值函数、ReLU 函数在原点处）。

### 直观理解次微分：从几何到优化
#### 1. 几何视角：支撑超平面的集合
- **可导点（如光滑凸函数 $f(x) = x^2$ 在 $x=1$ 处）**： 次微分 $\partial f(1) = \{2\}$（唯一支撑超平面，即切线，斜率为 2，与梯度一致）。 **直观**：光滑曲面的 “唯一切线”，次微分只有一个元素（梯度）。
- **不可导点（如 $f(x) = |x|$ 在 $x=0$ 处）**： 次微分 $\partial f(0) = [-1, 1]$，表示所有斜率在 $[-1, 1]$ 之间的直线都能支撑 $|x|$ 的图像（如 $y=x$、$y=-x$ 及中间直线）。 **直观**：“V” 型顶点处，有多条直线从下方支撑（不穿过图像），次微分是这些直线斜率的集合。
#### 2. 优化视角：近似导数的 “范围”
- **次梯度法**：在不可导点（如 LASSO 的 $\ell_1$ 范数、ReLU），用次微分（非空）迭代。选次梯度 $g \in \partial f(x)$，沿 $-g$ 方向更新，保证 $f(x - \alpha g) \geq f(x) - \alpha \|g\|^2$（函数值不增）。 **直观**：不可导点的 “导数” 不唯一，次微分给出所有可能的 “下降方向候选”，算法任选其一，仍能下降（如在 “V” 顶点，向左、向右或中间走，只要在次微分范围内，函数值不会上升）。
#### 3. 类比：楼梯与斜坡
- **可导点（斜坡）**：次微分是 “唯一坡度”（梯度），沿此走必下降。
- **不可导点（楼梯顶点）**：次微分是 “坡度范围”（如楼梯拐角，左、右、中间坡度都支撑），算法选任一坡度（次梯度），仍能沿支撑线方向下降。
#### 4. 高维示例：ReLU 函数 $f(x, y) = \max(0, x+y)$ 在 $(0,0)$ 处
次微分是 $(g_1, g_2)$ 满足 $g_1 + g_2 \leq 1$ 且 $g_1, g_2 \geq 0$（二维支撑区域），反映高维不可导点的支撑多样性。
### 总结
次微分是凸函数在某点处**所有支撑超平面斜率（或梯度向量）的集合**。可导点唯一（梯度），不可导点为区间 / 集合（支撑多样性）。它为不可导凸函数优化提供 “广义导数”，像在 “V” 顶点处，虽无唯一切线，但有众多支撑线引导下降，是次梯度法的理论基础。
**一句话记忆**：次微分是凸函数在该点 “下方所有支撑线的斜率集合”，可导时缩成一个点（梯度），不可导时展开成区间（如 “V” 顶点的斜率范围）。