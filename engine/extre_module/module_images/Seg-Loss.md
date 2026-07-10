## Seg损失函数公式总结

### 1. **总体掩码损失 (Mask Loss)**

掩码损失由两部分组成：

\[
\mathcal{L}_{\text{mask}} = \lambda_{\text{ce}} \mathcal{L}_{\text{ce}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}
\]

其中 \(\lambda_{\text{ce}}\) 和 \(\lambda_{\text{dice}}\) 是平衡系数。

---

### 2. **点采样策略 (Point Sampling)**

采用 **PointRend** 的不确定性采样策略，总共采样 \(P\) 个点：

- **不确定性采样点数**: \(P_{\text{uncertain}} = \lfloor \alpha \cdot P \rfloor\)
- **随机采样点数**: \(P_{\text{random}} = P - P_{\text{uncertain}}\)

其中 \(\alpha = 0.75\) 是重要性采样比例，过采样比例为 3。

**不确定性计算**：
\[
U(\mathbf{z}) = -|\mathbf{z}|
\]

其中 \(\mathbf{z}\) 是采样点的 logit 预测值。选择不确定性最高的 \(P_{\text{uncertain}}\) 个点。

---

### 3. **二值交叉熵损失 (Binary Cross-Entropy Loss)**

\[
\mathcal{L}_{\text{ce}} = \frac{1}{N} \sum_{i=1}^{N} \text{BCE}(\mathbf{p}_i, \mathbf{y}_i)
\]

其中：
\[
\text{BCE}(\mathbf{p}_i, \mathbf{y}_i) = -\frac{1}{P}\sum_{j=1}^{P} \left[ y_{ij} \log(\sigma(p_{ij})) + (1 - y_{ij}) \log(1 - \sigma(p_{ij})) \right]
\]

- \(N\) 是匹配的实例数量
- \(P\) 是采样点数量
- \(\mathbf{p}_i \in \mathbb{R}^P\) 是第 \(i\) 个实例在采样点的 logit 预测
- \(\mathbf{y}_i \in \{0, 1\}^P\) 是对应的真值标签
- \(\sigma(\cdot)\) 是 sigmoid 函数

---

### 4. **Dice 损失 (Dice Loss)**

\[
\mathcal{L}_{\text{dice}} = \frac{1}{N} \sum_{i=1}^{N} \left(1 - \frac{2\sum_{j=1}^{P} \hat{y}_{ij} \cdot y_{ij} + 1}{\sum_{j=1}^{P} \hat{y}_{ij} + \sum_{j=1}^{P} y_{ij} + 1}\right)
\]

其中：
- \(\hat{y}_{ij} = \sigma(p_{ij})\) 是第 \(i\) 个实例在第 \(j\) 个采样点的预测概率
- \(y_{ij} \in \{0, 1\}\) 是对应的真值标签
- 分子分母都加 1 用于数值稳定性（平滑项）

---

### 5. **双线性插值采样 (Bilinear Interpolation)**

对于归一化坐标 \(\mathbf{c} \in [0, 1]^2\)，通过以下变换进行网格采样：

\[
\mathbf{c}' = 2\mathbf{c} - 1
\]

将坐标从 \([0, 1]\) 映射到 \([-1, 1]\)，以适配 PyTorch 的 `grid_sample` 函数。

---

## 论文描述建议

在论文中可以这样描述：

> **Mask Loss with Point Sampling.** Following PointRend [1], we supervise mask predictions using point-based sampling to improve efficiency. For each matched instance, we sample \(P\) points, where 75% are selected based on prediction uncertainty \(U(\mathbf{z}) = -|\mathbf{z}|\), and 25% are sampled uniformly at random. The mask loss combines binary cross-entropy and Dice loss:
>
> \[
> \mathcal{L}_{\text{mask}} = \lambda_{\text{ce}} \mathcal{L}_{\text{ce}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}
> \]
>
> where \(\mathcal{L}_{\text{ce}}\) is the sigmoid binary cross-entropy loss and \(\mathcal{L}_{\text{dice}}\) is the Dice loss computed on the sampled points. Both losses are normalized by the total number of matched instances across the batch.

---

**参考文献**:
[1] Kirillov, A., Wu, Y., He, K., & Girshick, R. (2020). PointRend: Image segmentation as rendering. In CVPR.