# 头相关传输函数（HRTF）综述

## 0. 符号约定与基础概念

在阅读本文前，请先熟悉以下核心符号与概念：

### 0.1 坐标与方向符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $\mathbf{r} = (x, y, z)$ | 空间位置矢量 | 笛卡尔坐标系中的三维位置 |
| $r = \|\mathbf{r}\|$ | 径向距离 | 从原点到空间点的距离 |
| $\theta$ | 方位角（Azimuth） | 水平面内的角度，通常 $\theta \in [-\pi, \pi]$，正前方为 $0$ |
| $\phi$ | 仰角（Elevation） | 垂直方向角度，$\phi \in [-\pi/2, \pi/2]$，水平面为 $0$ |
| $\Omega = (\theta, \phi)$ | 空间方向 | 球坐标中的方向二元组，表示声源相对于听者的入射方向 |

### 0.2 频域与时域符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $t$ | 时间 | 单位：秒 (s) |
| $\omega = 2\pi f$ | 角频率 | 单位：弧度/秒 (rad/s)，$f$ 为频率 (Hz) |
| $k = \omega/c = 2\pi/\lambda$ | 波数 | 单位：弧度/米 (rad/m)，$\lambda$ 为波长 |
| $c$ | 声速 | 空气中约 $343$ m/s（20°C） |
| $j = \sqrt{-1}$ | 虚数单位 | 工程中常用 $j$ 代替数学中的 $i$ |

### 0.3 声压与传递函数符号

| 符号 | 含义 | 说明 |
|------|------|------|
| $p(\mathbf{r}, t)$ | 时域声压 | 空间点 $\mathbf{r}$ 处、时刻 $t$ 的声压，单位：帕斯卡 (Pa) |
| $p(\mathbf{r}, \omega)$ | 频域声压 | $p(\mathbf{r}, t)$ 的傅里叶变换，是复数 |
| $P_{ear}(\omega, \Omega)$ | 耳道入口声压频谱 | 方向 $\Omega$ 入射时耳道处的声压 |
| $S(\omega, \Omega)$ | 参考声压频谱 | 自由场中参考点（通常为头部中心）的声压 |
| $H_{ear}(\omega, \Omega)$ | HRTF | 头相关传输函数，复数值 |
| $h_{ear}(t, \Omega)$ | HRIR | 头相关脉冲响应，HRTF 的时域形式 |


---

## 1. 背景与基本定义

### 1.1 物理意义

**头相关传输函数（Head-Related Transfer Function, HRTF）** 描述了声源在空间方向 $\Omega = (\theta, \phi)$ 入射时，外耳道入口处声压与自由场声压之间的频率响应关系。

它综合刻画了以下物理效应：
- **头部散射**：声波遇到头部时发生绑射和反射
- **躯干反射**：肩部和躯干对声波的反射
- **耳廓衍射**：耳廓复杂褶皱结构对声波的多路径干涉

### 1.2 数学定义（频域）

$$
H_{ear}(\omega, \Omega) = \frac{P_{ear}(\omega, \Omega)}{S(\omega, \Omega)}
$$

**各符号详解**：
- $H_{ear}(\omega, \Omega)$：HRTF，是关于角频率 $\omega$ 和方向 $\Omega$ 的复值函数
  - 下标 $ear$ 表示左耳 ($L$) 或右耳 ($R$)
  - 复数形式 $H = |H|e^{j\angle H}$ 包含幅度 $|H|$ 和相位 $\angle H$
- $P_{ear}(\omega, \Omega)$：耳道入口处声压的傅里叶变换
- $S(\omega, \Omega)$：参考点声压的傅里叶变换，通常取头部中心位置在无头部存在时的自由场声压

**物理解释**：HRTF 本质上是一个线性系统的传递函数，描述了"自由场声源"到"耳道入口"这一声学路径的频率响应特性。

---

## 2. 声学建模框架

### 2.1 几何与物理假设

**问题设定**：
1. 头部建模为具有复杂边界的三维散射体 $\mathcal{B}$
2. 外部介质为空气，声速 $c \approx 343$ m/s
3. 假设声场为线性、时不变系统

### 2.2 Helmholtz 方程详解

**时域波动方程**：

声波在均匀介质中的传播满足波动方程：
$$
\nabla^2 p(\mathbf{r}, t) - \frac{1}{c^2} \frac{\partial^2 p(\mathbf{r}, t)}{\partial t^2} = 0
$$

其中：
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2}$ 是**拉普拉斯算子**（Laplacian），描述空间二阶导数
- $p(\mathbf{r}, t)$ 是时域声压场
- $c$ 是声速

**频域变换推导**：

假设声场为单频稳态（或对任意信号做傅里叶分解），令：
$$
p(\mathbf{r}, t) = \text{Re}\left[ p(\mathbf{r}, \omega) e^{j\omega t} \right]
$$

代入波动方程：
$$
\nabla^2 \left[ p(\mathbf{r}, \omega) e^{j\omega t} \right] - \frac{1}{c^2} \frac{\partial^2}{\partial t^2} \left[ p(\mathbf{r}, \omega) e^{j\omega t} \right] = 0
$$

由于 $\frac{\partial^2}{\partial t^2} e^{j\omega t} = (j\omega)^2 e^{j\omega t} = -\omega^2 e^{j\omega t}$，得到：
$$
\nabla^2 p(\mathbf{r}, \omega) e^{j\omega t} + \frac{\omega^2}{c^2} p(\mathbf{r}, \omega) e^{j\omega t} = 0
$$

消去公因子 $e^{j\omega t}$，定义波数 $k = \omega/c$，得到 **Helmholtz 方程**：
$$
\boxed{\nabla^2 p(\mathbf{r}, \omega) + k^2 p(\mathbf{r}, \omega) = 0}
$$

**物理意义**：Helmholtz 方程是波动方程在频域的等价形式，描述了单一频率成分的空间分布。


### 2.3 Neumann 边界条件详解

**刚性边界假设**：

假设头部表面为声学刚性（即声波无法穿透），则边界上的法向速度为零。根据欧拉方程（动量守恒）：
$$
\rho_0 \frac{\partial \mathbf{v}}{\partial t} = -\nabla p
$$

其中 $\rho_0$ 为空气密度，$\mathbf{v}$ 为质点速度。法向速度为零意味着：
$$
v_n = -\frac{1}{j\omega\rho_0} \frac{\partial p}{\partial n} = 0
$$

因此得到 **Neumann 边界条件**：
$$
\boxed{\left. \frac{\partial p}{\partial n} \right|_{\mathbf{r} \in \partial\mathcal{B}} = 0}
$$

**符号说明**：
- $\partial\mathcal{B}$：散射体 $\mathcal{B}$ 的边界表面（即头部表面）
- $\frac{\partial p}{\partial n}$：声压沿边界外法向 $\mathbf{n}$ 的方向导数
- 该条件表示声波在刚性表面上完全反射，无能量透射

### 2.4 入射场与散射场分解

**总场分解**：

当声波遇到散射体时，总声场可分解为两部分：
$$
\boxed{p(\mathbf{r}, \omega, \Omega) = p^{\text{inc}}(\mathbf{r}, \omega, \Omega) + p^{\text{scat}}(\mathbf{r}, \omega, \Omega)}
$$

**各项含义**：

| 符号 | 名称 | 物理意义 |
|------|------|----------|
| $p(\mathbf{r}, \omega, \Omega)$ | 总声压场 | 实际测量到的声压 |
| $p^{\text{inc}}(\mathbf{r}, \omega, \Omega)$ | 入射场 | 假设无散射体存在时的声场（自由场） |
| $p^{\text{scat}}(\mathbf{r}, \omega, \Omega)$ | 散射场 | 由散射体引起的附加声场 |

**入射场的典型形式**：

对于从方向 $\Omega$ 入射的平面波：
$$
p^{\text{inc}}(\mathbf{r}, \omega, \Omega) = A(\omega) e^{-j\mathbf{k} \cdot \mathbf{r}}
$$

其中：
- $A(\omega)$ 是声源的频谱幅度
- $\mathbf{k} = k \hat{\mathbf{d}}(\Omega)$ 是波矢量，$\hat{\mathbf{d}}(\Omega)$ 是入射方向的单位矢量
- $\mathbf{k} \cdot \mathbf{r} = k(x\sin\theta\cos\phi + y\sin\theta\sin\phi + z\cos\phi)$

**HRTF 的物理解释**：

耳道入口处的 HRTF 即为该点总声压与入射声压的比值：
$$
H_{ear}(\omega, \Omega) = \frac{p(\mathbf{r}_{ear}, \omega, \Omega)}{p^{\text{inc}}(\mathbf{r}_{ref}, \omega, \Omega)} = 1 + \frac{p^{\text{scat}}(\mathbf{r}_{ear}, \omega, \Omega)}{p^{\text{inc}}(\mathbf{r}_{ref}, \omega, \Omega)}
$$

---

### 2.5 HRTF 与 HRIR 的关系

**傅里叶变换对**：

HRTF 是频域函数，其时域对应为 **头相关脉冲响应（HRIR）**：
$$
h_{ear}(t, \Omega) = \mathcal{F}^{-1}\{H_{ear}(\omega, \Omega)\} = \frac{1}{2\pi} \int_{-\infty}^{\infty} H_{ear}(\omega, \Omega) e^{j\omega t} d\omega
$$

**卷积关系推导**：

设输入信号为 $x(t)$，其频谱为 $X(\omega) = \mathcal{F}\{x(t)\}$。

根据线性系统理论，耳道入口处的输出频谱为：
$$
Y_{ear}(\omega, \Omega) = H_{ear}(\omega, \Omega) \cdot X(\omega)
$$

对两边做逆傅里叶变换，利用卷积定理：
$$
\boxed{y_{ear}(t, \Omega) = (x * h_{ear})(t) = \int_{-\infty}^{\infty} x(\tau) h_{ear}(t-\tau, \Omega) d\tau}
$$

**实际意义**：任意声音信号 $x(t)$ 从方向 $\Omega$ 入射到耳朵时，只需与该方向的 HRIR 做卷积即可得到耳道入口处的信号。


### 2.6 球谐展开详解

**球坐标系定义**：

在球坐标 $(r, \theta, \phi)$ 中：
- $r$：径向距离
- $\theta$：方位角（水平面内）
- $\phi$：仰角（或极角）

**球谐函数 $Y_n^m(\Omega)$**：

球谐函数是球面上的正交基函数，定义为：
$$
Y_n^m(\theta, \phi) = \sqrt{\frac{(2n+1)}{4\pi} \frac{(n-m)!}{(n+m)!}} P_n^m(\cos\phi) e^{jm\theta}
$$

其中：
- $n = 0, 1, 2, \ldots$ 是阶数（order）
- $m = -n, -n+1, \ldots, n$ 是度数（degree）
- $P_n^m$ 是关联勒让德多项式

**球谐函数的正交性**：
$$
\int_{\mathbb{S}^2} Y_n^m(\Omega) \overline{Y_{n'}^{m'}(\Omega)} d\Omega = \delta_{nn'}\delta_{mm'}
$$

**声场的球谐展开**：

外部声场可用球谐函数展开：
$$
p(r, \Omega, \omega) = \sum_{n=0}^{\infty} \sum_{m=-n}^{n} a_{nm}(\omega) j_n(kr) Y_n^m(\Omega)
$$

**符号说明**：
- $a_{nm}(\omega)$：球谐系数，由声源特性决定
- $j_n(kr)$：第一类球贝塞尔函数，描述径向依赖性
- $Y_n^m(\Omega)$：球谐函数，描述角度依赖性

**HRTF 的球谐表示**：

由于 HRTF 是定义在球面上的函数，可展开为：
$$
\boxed{H_{ear}(\omega, \Omega) = \sum_{n=0}^{N} \sum_{m=-n}^{n} \alpha_{nm}^{ear}(\omega) Y_n^m(\Omega)}
$$

其中：
- $\alpha_{nm}^{ear}(\omega)$：HRTF 的球谐系数
- $N$：截断阶数，决定空间分辨率
- 总系数数量为 $(N+1)^2$

**截断阶数的选择**：
- $N = 4$：$(4+1)^2 = 25$ 个系数，适合低频
- $N = 10$：$(10+1)^2 = 121$ 个系数，中等精度
- $N = 35$：$(35+1)^2 = 1296$ 个系数，高精度全频段

---

### 2.7 边界积分方程与数值解法

**格林函数**：

自由空间格林函数满足：
$$
\nabla^2 G(\mathbf{r}, \mathbf{r}_0) + k^2 G(\mathbf{r}, \mathbf{r}_0) = -\delta(\mathbf{r} - \mathbf{r}_0)
$$

其解为：
$$
G(\mathbf{r}, \mathbf{r}_0) = \frac{e^{-jk|\mathbf{r} - \mathbf{r}_0|}}{4\pi|\mathbf{r} - \mathbf{r}_0|}
$$

**物理意义**：$G(\mathbf{r}, \mathbf{r}_0)$ 表示位于 $\mathbf{r}_0$ 的点源在 $\mathbf{r}$ 处产生的声压。

**格林第二恒等式**：

对于满足 Helmholtz 方程的两个函数 $p$ 和 $G$，有：
$$
\int_V (p\nabla^2 G - G\nabla^2 p) dV = \oint_{\partial V} \left( p\frac{\partial G}{\partial n} - G\frac{\partial p}{\partial n} \right) dS
$$

**边界积分方程推导**：

将格林函数代入，利用 $\nabla^2 G = -k^2 G - \delta(\mathbf{r} - \mathbf{r}_0)$，得到：
$$
\boxed{\frac{1}{2} p(\mathbf{r}_0) = p^{\text{inc}}(\mathbf{r}_0) + \int_{\partial\mathcal{B}} \left[ G(\mathbf{r}, \mathbf{r}_0) \frac{\partial p}{\partial n}(\mathbf{r}) - p(\mathbf{r}) \frac{\partial G}{\partial n}(\mathbf{r}, \mathbf{r}_0) \right] dS}
$$

**符号说明**：
- $\mathbf{r}_0$：场点（求解声压的位置）
- $\mathbf{r}$：边界上的积分点
- $\frac{1}{2}$ 因子：当 $\mathbf{r}_0$ 在光滑边界上时出现
- 对于刚性边界，$\frac{\partial p}{\partial n} = 0$，方程简化

**边界元法（BEM）**：

将边界离散为 $N_e$ 个单元，得到线性系统：
$$
\mathbf{A}\mathbf{p} = \mathbf{b}
$$

其中：
- $\mathbf{A}$：$N_e \times N_e$ 系数矩阵，由格林函数积分构成
- $\mathbf{p}$：边界上的声压向量
- $\mathbf{b}$：入射场贡献


### 2.8 耳道终端与鼓膜阻抗

**一维波导模型**：

耳道可近似为长度 $L$（约 25-30 mm）的圆柱形波导。设耳道入口（EDE）声压为 $P_{ede}$，鼓膜处声压为 $P_{tm}$。

**传输线方程**：

根据一维声学传输线理论：
$$
\begin{pmatrix} P_{tm} \\ U_{tm} \end{pmatrix} = \begin{pmatrix} \cos(kL) & jZ_0\sin(kL) \\ \frac{j}{Z_0}\sin(kL) & \cos(kL) \end{pmatrix} \begin{pmatrix} P_{ede} \\ U_{ede} \end{pmatrix}
$$

其中：
- $Z_0 = \rho_0 c / S$：波导特性阻抗，$S$ 为耳道截面积
- $U$：体积速度
- $k = \omega/c$：波数
- $L$：耳道有效长度

**鼓膜处声压**：

假设鼓膜阻抗为 $Z_{tm}$，则 $P_{tm} = Z_{tm} U_{tm}$，代入得：
$$
\boxed{P_{tm}(\omega) = P_{ede}(\omega) \cos(kL) - j Z_0 U_{ede}(\omega) \sin(kL)}
$$

---

## 3. 典型推导与近似

### 3.1 刚性球模型详细推导

**问题设定**：
- 头部近似为半径 $a$（约 8.75 cm）的刚性球体
- 耳朵位于球面上某点
- 入射波为平面波

**球坐标中的 Helmholtz 方程**：

在球坐标中，Helmholtz 方程为：
$$
\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 \frac{\partial p}{\partial r}\right) + \frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta \frac{\partial p}{\partial \theta}\right) + \frac{1}{r^2\sin^2\theta}\frac{\partial^2 p}{\partial \phi^2} + k^2 p = 0
$$

**分离变量解**：

设 $p(r, \theta, \phi) = R(r) \Theta(\theta) \Phi(\phi)$，可得：
- 径向方程的解：球贝塞尔函数 $j_n(kr)$ 和球汉克尔函数 $h_n^{(2)}(kr)$
- 角度方程的解：球谐函数 $Y_n^m(\theta, \phi)$

**入射平面波展开**：

沿 $z$ 轴入射的平面波可展开为：
$$
p^{\text{inc}} = e^{jkz} = e^{jkr\cos\theta} = \sum_{n=0}^{\infty} (2n+1) j^n j_n(kr) P_n(\cos\theta)
$$

其中 $P_n$ 是勒让德多项式。

**散射场形式**：

散射场必须满足辐射条件（向外传播），因此用球汉克尔函数：
$$
p^{\text{scat}} = \sum_{n=0}^{\infty} A_n h_n^{(2)}(kr) P_n(\cos\theta)
$$

**边界条件求解系数**：

在球面 $r = a$ 上应用 Neumann 条件 $\frac{\partial p}{\partial r}\big|_{r=a} = 0$：
$$
\frac{\partial}{\partial r}\left[ (2n+1)j^n j_n(kr) + A_n h_n^{(2)}(kr) \right]_{r=a} = 0
$$

解得：
$$
A_n = -(2n+1)j^n \frac{j_n'(ka)}{h_n^{(2)'}(ka)}
$$

**球面上的 HRTF**：

在球面 $r = a$ 上，总声压为：
$$
\boxed{H(\omega, \theta) = \sum_{n=0}^{\infty} (2n+1) j^n \left[ j_n(ka) - \frac{j_n'(ka)}{h_n^{(2)'}(ka)} h_n^{(2)}(ka) \right] P_n(\cos\theta)}
$$

**符号说明**：
- $j_n(x)$：第一类球贝塞尔函数
- $h_n^{(2)}(x) = j_n(x) - jy_n(x)$：第二类球汉克尔函数
- $j_n'(x), h_n^{(2)'}(x)$：对应函数的导数
- $P_n(\cos\theta)$：$n$ 阶勒让德多项式
- $\theta$：声源方向与耳朵位置的夹角


### 3.2 ITD 与 ILD 近似推导

#### 3.2.1 双耳时差（ITD）

**简单几何模型**：

假设头部为直径 $d$ 的球体，双耳位于球体两侧。对于方位角 $\theta$ 的声源：

```
        声源
          \
           \  θ
            \
    左耳 ●---●---● 右耳
            d
```

**路径差计算**：

声波到达两耳的路径差为：
$$
\Delta L = d \sin\theta
$$

因此时间差为：
$$
\boxed{\text{ITD}(\theta) \approx \frac{d}{c} \sin\theta}
$$

其中 $d \approx 0.18$ m（双耳间距）。

**Woodworth-Green 模型**：

考虑声波需要绑射绕过头部，对于耳侧声源更准确：
$$
\boxed{\text{ITD}(\theta) = \frac{a}{c} \left( \sin\theta + \theta \right), \quad |\theta| \leq \frac{\pi}{2}}
$$

**推导过程**：
1. 对于近耳：声波直接到达，路径 $= a\sin\theta$
2. 对于远耳：声波需绕过头部弧线，额外路径 $= a\theta$
3. 总路径差 $= a(\sin\theta + \theta)$

其中 $a \approx 0.0875$ m 为等效头部半径。

#### 3.2.2 双耳级差（ILD）

**定义**：

ILD 是左右耳 HRTF 幅度的比值（以 dB 表示）：
$$
\boxed{\text{ILD}(\omega, \theta) = 20 \log_{10} \left| \frac{H_L(\omega, \theta)}{H_R(\omega, \theta)} \right|}
$$

**物理机制**：
- 低频（$< 500$ Hz）：波长远大于头部尺寸，ILD 很小
- 高频（$> 1500$ Hz）：头部产生明显的声影效应，ILD 可达 20 dB

**频率依赖性**：

ILD 随频率增加而增大，近似关系：
$$
\text{ILD}(\omega, \theta) \propto (ka)^2 \sin\theta, \quad \text{当 } ka < 1
$$

---

### 3.3 最小相位与纯延迟分解

**分解形式**：

HRTF 可分解为：
$$
\boxed{H_{ear}(\omega, \Omega) = H_{\text{min}}(\omega, \Omega) \cdot e^{-j\omega\tau(\Omega)}}
$$

**各项含义**：
- $H_{\text{min}}(\omega, \Omega)$：最小相位部分，其相位完全由幅度谱决定
- $e^{-j\omega\tau(\Omega)}$：纯延迟（全通）部分
- $\tau(\Omega)$：方向相关的时延，约等于 ITD

**最小相位系统的性质**：

最小相位系统的相位与幅度通过 Hilbert 变换关联：
$$
\angle H_{\text{min}}(\omega) = -\mathcal{H}\{\log|H(\omega)|\}
$$

其中 Hilbert 变换定义为：
$$
\mathcal{H}\{f(\omega)\} = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{f(\omega')}{\omega - \omega'} d\omega'
$$

**实际意义**：
- 最小相位滤波器具有最短的群延迟
- 分解后可用更短的 FIR 滤波器实现 HRTF
- 纯延迟部分可单独处理，提高计算效率

---

### 3.4 耳廓衍射与谱凹点

**多路径干涉模型**：

耳廓的复杂褶皱结构使声波通过多条路径到达耳道入口：

```
声源 ──→ 直接路径 ──→ 耳道
    └──→ 反射路径 ──→ 耳道
         (经耳廓反射)
```

**谱凹点形成原理**：

设直接路径和反射路径的长度差为 $\Delta\ell$，当两路径相位差为 $\pi$（反相）时发生相消干涉：
$$
k \cdot \Delta\ell = (2m+1)\pi, \quad m = 0, 1, 2, \ldots
$$

即：
$$
\frac{2\pi f}{c} \cdot \Delta\ell = (2m+1)\pi
$$

解得谱凹点频率：
$$
\boxed{f_{\text{notch}} = \frac{(2m+1)c}{2\Delta\ell} \approx \frac{c}{2\Delta\ell}} \quad (\text{取 } m=0)
$$

**数值示例**：
- 若 $\Delta\ell = 2$ cm，$c = 343$ m/s
- $f_{\text{notch}} = \frac{343}{2 \times 0.02} = 8575$ Hz

**空间定位意义**：
- 谱凹点频率随声源仰角变化
- 提供前后、上下方向的分辨线索
- 是 HRTF 个性化的关键特征


---

## 4. 测量与估计方法

### 4.1 直接测量

1. **自由场测量**：将高密度扬声器阵列围绕受试者，逐方向播放扫频或 MLS 信号
2. **探针麦克风**：将微型麦克风置于耳道入口，记录响应
3. **参考去卷积**：以已知刺激信号 $s(t)$ 与记录 $y(t)$ 为基础，通过去卷积求得 HRIR

**去卷积原理**：

设系统输入为 $s(t)$，输出为 $y(t)$，则：
$$
Y(\omega) = H(\omega) \cdot S(\omega)
$$

HRTF 可通过频域除法得到：
$$
H(\omega) = \frac{Y(\omega)}{S(\omega)}
$$

### 4.2 激励信号选择

- **最大长度序列（MLS）**：利用伪随机序列的循环自相关特性，适合快速测量
- **指数正弦扫频（ESS）**：抗非线性能力强，常用于精确测量

### 4.3 个性化估计

- **参数化建模**：用主成分分析（PCA）或球谐系数表征个体差异
- **机器学习回归**：输入体测数据（头围、耳廓形状等）预测 HRTF
- **深度学习**：以少量校准方向为监督，通过神经场或扩散模型生成全向 HRTF

### 4.4 典型测量流程

1. **坐标校准**：利用激光定位或摄影测量确保头部中心与坐标原点重合
2. **系统标定**：逐个扬声器播放参考信号，测得放大器与声卡响应，生成逆滤波器
3. **序列采集**：按照球面测点顺序播放激励（MLS/ESS），记录耳道信号
4. **后处理**：窗函数截取、平均多次测量、噪声门控、参考信号去卷积，导出 SOFA 文件

---

## 5. 建模与压缩技术

### 5.1 球谐插值

**最小二乘拟合**：

给定 $M$ 个测量方向 $\{\Omega_1, \ldots, \Omega_M\}$ 上的 HRTF 值 $\mathbf{h}(\omega) = [H(\omega, \Omega_1), \ldots, H(\omega, \Omega_M)]^T$，构建球谐矩阵：
$$
\mathbf{Y} = \begin{pmatrix}
Y_0^0(\Omega_1) & Y_1^{-1}(\Omega_1) & \cdots & Y_N^N(\Omega_1) \\
Y_0^0(\Omega_2) & Y_1^{-1}(\Omega_2) & \cdots & Y_N^N(\Omega_2) \\
\vdots & \vdots & \ddots & \vdots \\
Y_0^0(\Omega_M) & Y_1^{-1}(\Omega_M) & \cdots & Y_N^N(\Omega_M)
\end{pmatrix}
$$

球谐系数通过最小二乘求解：
$$
\boxed{\boldsymbol{\alpha}(\omega) = (\mathbf{Y}^H \mathbf{Y})^{-1} \mathbf{Y}^H \mathbf{h}(\omega)}
$$

其中 $\mathbf{Y}^H$ 表示共轭转置。

### 5.2 PCA/ICA 表征

**PCA 分解**：

将对数幅度谱 $\log|H_{ear}(\omega, \Omega)|$ 构建为高维向量，通过 PCA 得到：
$$
\boxed{\mathbf{h}_{ear} \approx \mathbf{U}\mathbf{w} + \boldsymbol{\mu}}
$$

其中：
- $\boldsymbol{\mu}$：所有个体的平均 HRTF
- $\mathbf{U}$：主成分基矩阵（正交）
- $\mathbf{w}$：个体特定的权重向量

### 5.3 数据驱动模型

- **神经辐射场（NeRF）/SIREN**：以方向与频率为输入，输出复数 HRTF
- **扩散模型**：生成逼真个性化 HRTF 样本

### 5.4 最小相位与群延迟拟合

**群延迟定义**：
$$
\tau_g(\omega) = -\frac{d}{d\omega} \angle H(\omega)
$$

**最小相位重建**：
$$
\boxed{\angle H_{\text{min}}(\omega) = -\mathcal{H}\{\log|H(\omega)|\}}
$$

---

## 6. 插值与外推

### 6.1 方向插值

利用球谐或分段样条在球面上插值。

### 6.2 频率插值

对数幅度与群延迟分别插值以避免相位跳变。

### 6.3 近场到远场外推

基于声源距离 $r$ 的 Green 函数：
$$
\boxed{H(\omega, r, \Omega) = H_{\infty}(\omega, \Omega) \frac{e^{-jkr}}{r}}
$$

其中 $H_{\infty}$ 为远场 HRTF。

### 6.4 距离插值

在同一方向测得多个距离样本后，可对 $H(\omega, r)$ 进行有理函数或对数线性拟合。


---

## 7. 应用场景

### 7.1 双耳渲染与虚拟现实

**基本原理**：

将单声道信号 $x(t)$ 经过左右耳 HRTF 卷积后合成双声道：
$$
\begin{aligned}
y_L(t) &= x(t) * h_L(t, \Omega) \\
y_R(t) &= x(t) * h_R(t, \Omega)
\end{aligned}
$$

**典型渲染管线**：
1. 根据头部追踪数据更新声源方向 $\Omega(t)$
2. 通过球谐或分段加权插值得到当前方向 HRTF
3. 使用 FFT Overlap-Add 快速卷积生成左右耳信号
4. 混合环境混响或早期反射

### 7.2 语音增强与波束形成

**MVDR 波束形成器**：

使用 HRTF 作为引导矢量：
$$
\boxed{\mathbf{w}_{\text{MVDR}} = \frac{\mathbf{R}_{nn}^{-1} \mathbf{h}_{\text{HRTF}}}{\mathbf{h}_{\text{HRTF}}^H \mathbf{R}_{nn}^{-1} \mathbf{h}_{\text{HRTF}}}}
$$

**符号说明**：
- $\mathbf{w}_{\text{MVDR}}$：波束形成器权重向量
- $\mathbf{R}_{nn}$：噪声协方差矩阵
- $\mathbf{h}_{\text{HRTF}}$：麦克风阵列对耳道的传递向量
- $(\cdot)^H$：共轭转置

**实现步骤**：
1. 估计噪声协方差 $\mathbf{R}_{nn}$ 与目标协方差 $\mathbf{R}_{ss}$
2. 由阵列几何与 HRTF 推出目标引导矢量
3. 设计滤波器（MVDR/LCMV/GEV）
4. 结合交叉谈话消除（XTC）确保信号准确传递

### 7.3 说话人定位

**最大似然定位**：
$$
\boxed{\hat{\Omega} = \arg\max_{\Omega} p(\mathbf{f}_{\text{obs}} | H(\omega, \Omega))}
$$

其中 $\mathbf{f}_{\text{obs}}$ 包括 GCC-PHAT、ILD、谱凹点等特征。

### 7.4 助听与可穿戴设备

- 用个性化 HRTF 增强方位感
- 改善多说话人场景的可懂度
- 自适应助听器可在定向麦克风与 HRTF 滤波之间切换

---

## 8. 实践建议

1. **频率采样**：常用 0–24 kHz，保证 48 kHz 采样率的 Nyquist 要求
2. **方向分辨率**：全向测量至少 710+ 个方向以避免空间混叠
3. **噪声与非线性校正**：测量前后需进行噪声门控、系统响应补偿
4. **数据格式**：SOFA (Spatially Oriented Format for Acoustics) 是交换标准

---

## 9. 评估指标

**幅度误差**：
$$
\boxed{E_{\text{mag}} = \sqrt{\frac{1}{K} \sum_{k=1}^{K} \left(|H_{\text{pred}}(\omega_k)| - |H_{\text{ref}}(\omega_k)|\right)^2}}
$$

**ITD/ILD 误差**：比较预测与测量的 ITD、ILD 曲线。

**感知评价**：通过 Localization Blur、外化评分或双耳听辨实验验证主观效果。

---

## 10. 数据集与工具

| 数据集 | 受试者数 | 方向数 | 采样率 | 特点 |
|--------|----------|--------|--------|------|
| CIPIC | 45 | 1250 | 44.1 kHz | 含体测参数 |
| ARI | 多个 | 多种 | 多种 | 真实耳与人工耳 |
| SCUT KEMAR | 1 (机器人头) | 高密度 | 48 kHz | 常用于仿真 |

**常用软件**：SOFA API、pysofaconventions、SPATIAL AUDIO WORKSTATION

---

## 11. 进一步阅读

- W. M. Hartmann, *Principles of Binaural Room Acoustics*
- H. Wierstorf et al., "A Free Database of Head-Related Impulse Response Measurements," AES
- SOFA Convention: [https://www.sofaconventions.org](https://www.sofaconventions.org)

---

## 附录 A：常用数学函数

### A.1 球贝塞尔函数

第一类球贝塞尔函数：
$$
j_n(x) = \sqrt{\frac{\pi}{2x}} J_{n+1/2}(x)
$$

前几阶：
$$
\begin{aligned}
j_0(x) &= \frac{\sin x}{x} \\
j_1(x) &= \frac{\sin x}{x^2} - \frac{\cos x}{x}
\end{aligned}
$$

### A.2 球汉克尔函数

第二类球汉克尔函数：
$$
h_n^{(2)}(x) = j_n(x) - j y_n(x)
$$

其中 $y_n(x)$ 是第二类球贝塞尔函数。

### A.3 勒让德多项式

递推关系：
$$
(n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
$$

前几阶：
$$
\begin{aligned}
P_0(x) &= 1 \\
P_1(x) &= x \\
P_2(x) &= \frac{1}{2}(3x^2 - 1)
\end{aligned}
$$

### A.4 球谐函数

实数形式的球谐函数：
$$
Y_n^m(\theta, \phi) = \begin{cases}
\sqrt{2} N_n^m P_n^m(\cos\phi) \cos(m\theta), & m > 0 \\
N_n^0 P_n^0(\cos\phi), & m = 0 \\
\sqrt{2} N_n^{|m|} P_n^{|m|}(\cos\phi) \sin(|m|\theta), & m < 0
\end{cases}
$$

其中归一化因子：
$$
N_n^m = \sqrt{\frac{(2n+1)(n-m)!}{4\pi(n+m)!}}
$$
