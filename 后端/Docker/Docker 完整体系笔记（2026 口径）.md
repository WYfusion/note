## 前置知识

> [!important]
> 
> 本笔记体系无前置页面依赖。建议读者具备基本的 Linux 命令行操作能力与网络基础概念。

---

## 0. 定位

> 本页是 Docker 完整体系笔记的 **L1 总纲**，负责提供全景鸟瞰、决策导向与子页面导航。不深入具体指令或配置细节，而是建立正确的心智模型（Mental Model）与学习路径。

---

## 1. Docker 解决的核心问题

> [!important]
> 
> 一句话：Docker 把「代码 + 运行时 + 系统依赖 + 启动命令」打包为**镜像（Image）**，在任何支持 Docker 的环境中以**容器（Container）**形式一致运行。

Docker 解决的不是「虚拟化」问题，而是**环境一致性 + 运行隔离 + 交付标准化**问题：

|核心问题|Docker 的回答|没有 Docker 时的典型痛点|
|---|---|---|
|**构建一致性**|Dockerfile 声明式定义环境|「我本地能跑，服务器上跑不了」|
|**运行隔离**|文件系统 / 网络 / 进程树 / 权限边界隔离|多服务互相污染、端口冲突、依赖版本打架|
|**交付一致性**|本地 / CI / 测试 / 生产同一镜像|手工部署、环境漂移、文档过期|
|**多服务编排**|Compose 声明拓扑 + 依赖 + 网络 + 存储|shell 脚本启动一堆进程、顺序出错就崩|

---

## 2. 四个最核心对象

![[Docker 完整体系笔记（2026 口径） - 2. 四个最核心对象 - 图 01 .excalidraw|800]]

|对象|本质|生命周期|关键认知|
|---|---|---|---|
|**镜像（Image）**|分层文件系统 + 元数据 + 默认启动配置|静态、不可变|尽量小、确定、可复现；不放运行态数据和机密|
|**容器（Container）**|镜像 + 可写层 + 运行时隔离|短暂、可删可重建|不是虚拟机；配置和数据必须外置|
|**卷/挂载（Volume / Mount）**|数据持久化与共享机制|1独立于容器生命周期|生产用 Volume、开发用 Bind、临时用 tmpfs|
|**网络（Network）**|容器间 / 容器与外部的通信边界|随项目创建销毁|默认用自定义 bridge；最少暴露端口|

---

## 3. 当前推荐总原则（2026 口径）

> [!important]
> 
> **工程判断优先级链**：
> 
> 1. 单服务定义 → **Dockerfile**
> 
> 1. 多服务协同 → **Compose**（`compose.yaml` 为主文件名）
> 
> 1. 数据持久化 → **Volume 优先**
> 
> 1. 开发态源码 → **Bind Mount**
> 
> 1. 网络 → **用户自定义 bridge**
> 
> 1. 安全 → **非 root / 最小权限 / 只读根文件系统**
> 
> 1. 构建 → **BuildKit 默认思维：多阶段 + 缓存 + secret mount**

---

## 4. 知识体系导航

/√以下是本笔记的完整子页面导航，按推荐阅读顺序编号：

|编号|主题|定位|
|---|---|---|
|§0|[[0 总体定位：建立正确心智模型]]|”建立 Image / Container / Volume / Network 的清晰边界认知|
|§1|[[1 Docker 基础对象：必须讲清的边界]]|Image / Container / daemon 的本质、生命周期与常见误区|
|§2|[[2 Dockerfile：当前应掌握的完整知识框架]]|指令体系、多阶段构建、BuildKit 高阶能力、最佳实践|
|§3|[[3 Docker Compose：当前推荐的项目主入口]]|Compose Specification、服务编排、多文件策略、开发增强|
|§4|[[4 挂载：Mount - Volume - Bind - tmpfs 一定要分清]]|(1三大挂载类型的本质区别、选型准则与设计原则|
|§5|[[5 网络：Docker 网络模式与选择准则]]|bridge / host / none / overlay / macvlan / ipvlan 的选择与隔离设计|
|§6|[[6 安全：当前项目设置中最关键的一层]]|运行时安全基线、rootless、机密管理、防火墙|
|§7|[[7 高阶构建能力：BuildKit - Cache - Bake - 多平台]]|BuildKit 思维、缓存优化、Bake 多镜像编排、多平台构建|
|§8|[[8 项目设置推荐：当前最稳的工程化组织方式]]|目录结构、Compose 文件组织、环境变量、数据目录|
|§9|[[9 开发 - 测试 - 生产 三态推荐]]|三种环境的 Docker 配置差异与推荐实践|
|§10|[[10 AI - Agent - LLM 项目中的 Docker 推荐做法]]|服务拆分、GPU、模型缓存、安全隔离的专项建议|
|§11–13|[[11 反模式清单 · 12 学习顺序 · 13 最终结论]]|高频踩坑、三层能力梯度、正确默认值汇总|

---

## 5. 学习路径决策树

![[Docker 完整体系笔记（2026 口径） - 5. 学习路径决策树 - 图 02 .excalidraw|800]]

---

## 延伸阅读

> [!important]
> 
> 子页面按编号顺序阅读即可覆盖完整体系。建议先通读 §0 ~ §5 建立基础，再按需深入 §6 ~ §10。

## 参考文献

- [1] Docker Official Documentation — [https://docs.docker.com/](https://docs.docker.com/)

- [2] Compose Specification — [https://docs.docker.com/compose/compose-file/](https://docs.docker.com/compose/compose-file/)

- [3] Docker Security Best Practices — [https://docs.docker.com/engine/security/](https://docs.docker.com/engine/security/)

- [4] BuildKit Documentation — [https://docs.docker.com/build/buildkit/](https://docs.docker.com/build/buildkit/)

[[0 总体定位：建立正确心智模型]]

[[1 Docker 基础对象：必须讲清的边界]]

[[2 Dockerfile：当前应掌握的完整知识框架]]

[[3 Docker Compose：当前推荐的项目主入口]]

[[5 网络：Docker 网络模式与选择准则]]

[[4 挂载：Mount - Volume - Bind - tmpfs 一定要分清]]

[[7 高阶构建能力：BuildKit - Cache - Bake - 多平台]]

[[6 安全：当前项目设置中最关键的一层]]

[[9 开发 - 测试 - 生产 三态推荐]]

[[8 项目设置推荐：当前最稳的工程化组织方式]]

[[11 反模式清单 · 12 学习顺序 · 13 最终结论]]

[[10 AI - Agent - LLM 项目中的 Docker 推荐做法]]