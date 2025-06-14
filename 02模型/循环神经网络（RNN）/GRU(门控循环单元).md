## 对比RNN
GRU和RNN的区别：
添加了两个门控单元：
    **更新门**：$$z_t​=σ(W_z​x_t​+U_z​h_{t−1​}+b_z​)$$
    **重置门**：$$r_t​=σ(W_r​x_t​+U_r​h_{t−1​}+b_r​)$$
    有助于捕捉序列中的短期依赖关系
    候选隐状态：$$\hat{h}_t​=tanh(W_h​x_t​+U_h​(r_t​⊙h_{t-1})+b_h​)$$ [[乘法#^8f925a|⊙]] 
    最终隐状态:$$h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde{h}_t$$
![[Pasted image 20250313200952.png|800]]
### 作用
更好的捕捉**长期依赖**关系，并减缓**梯度消失**问题。
- 重置门决定了**如何将新的输入信息与前面的记忆相结合**
- 更新门**定义了前面记忆保存到当前时间步的量**
- 如果将重置门设置为 1，更新门设置为 0，那么将再次获得标准 RNN 模型
- 若更新们$Z_t$一直设置为1，则隐层状态几乎不会发生变化，较早时刻的隐层信息一致被保留并传递到当前时间步。
- 对于重置门$R_t$中所有接近0的项， 候选隐状态是以$X_t$作为输入的多层感知机的结果。 因此，任何预先存在的隐状态都会被*重置*为默认值。