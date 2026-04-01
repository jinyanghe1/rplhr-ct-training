# RPLHR-CT 开发标准操作规程 (SOP)

> **版本**: v1.0  
> **生效日期**: 2026-04-01  
> **适用范围**: 所有代码修改和优化实验

---

## 一、核心原则

### 1.1 最小化增量原则

**定义**: 每次修改只涉及一个模块或一组相关参数，控制变更范围。

**具体要求**:
- ✅ **单一职责**: 一次commit只解决一个问题
- ✅ **限制变更数**: 一次commit不超过**3处**修改
- ✅ **原子性**: 每个commit可独立回滚，不依赖其他修改
- ✅ **清晰描述**: commit message必须说明修改了什么、为什么

**示例**:
```bash
# ✅ 好的commit - 单一修改
git commit -m "feat: 添加EAGLE Loss边缘感知损失

- 修改: code/loss.py 添加eagle_loss函数
- 验证: Epoch 5 PSNR从17.97提升到18.45 (+0.48dB)
- 回滚: 删除loss.py中的eagle_loss调用即可"

# ❌ 不好的commit - 多处修改混杂
git commit -m "update: 一些优化"
# (包含了EAGLE Loss + 数据增强 + dropout调整 + 学习率修改)
```

### 1.2 修改-验证-闭环原则

**定义**: 每个修改必须经过benchmark验证，证明有效才能保留。

**流程**:
```
修改代码 → 训练验证(5-10epoch) → Benchmark对比 → 决策
                                          ↓
                                    PSNR提升? 
                                          ↓
                              是 → 保留，继续训练
                              否 → 回滚，分析原因
```

**具体要求**:
- ✅ **基准对比**: 必须与原配置进行公平对比（相同epoch数）
- ✅ **统计显著性**: 至少验证3次，确保不是随机波动
- ✅ **回滚优先**: 无提升时**优先考虑回滚**，而非强行解释
- ✅ **严格证明**: 如不回滚，必须提供**定量证据**证明更优性

**Benchmark标准**:

| 指标 | 基线 | 修改后 | 决策 |
|------|------|--------|------|
| Val PSNR | 17.97 dB | > 18.2 dB (+0.23dB) | ✅ 保留 |
| Val PSNR | 17.97 dB | 17.80 dB (-0.17dB) | ❌ 回滚 |
| Val PSNR | 17.97 dB | 18.05 dB (+0.08dB) | ⚠️ 重新验证 |

**更优性证明标准**（不回滚时必须满足至少一条）:
1. **定量指标**: PSNR提升 > 0.2 dB，或SSIM提升 > 0.05
2. **收敛速度**: 达到相同PSNR所需epoch减少 > 20%
3. **稳定性**: 训练过程loss波动降低 > 30%
4. **定性分析**: 视觉质量显著改善（需side-by-side对比图）
5. **理论支撑**: 有文献证明该修改在类似任务中有效

---

## 二、实验流程规范

### 2.1 实验前准备

**必须记录**:
```markdown
## 实验记录模板

**实验ID**: EXP_YYYYMMDD_NNN
**日期**: YYYY-MM-DD
**修改内容**: 
- 文件: code/xxx.py
- 修改: 具体描述

**预期效果**: 
- 预期PSNR提升: +X dB
- 理论依据: 文献/推理

**基线配置**: 
- commit: abc1234
- 最佳PSNR: XX.XX dB @ Epoch X
```

### 2.2 实验执行

**快速验证**:
```bash
# 1. 创建新分支
git checkout -b exp/eagle_loss

# 2. 实施修改（最多3处）
vim code/loss.py

# 3. 本地测试（可选）
python -c "from loss import eagle_loss; print('OK')"

# 4. 提交修改
git add code/loss.py
git commit -m "feat: 添加EAGLE Loss (EXP_20240401_001)"

# 5. push并训练
git push origin exp/eagle_loss
ssh autodl "cd rplhr-ct && git pull && python train.py --epoch 10"
```

**训练监控**:
- 每5个epoch检查一次benchmark
- 记录training_history.json
- 关注Loss曲线是否异常

### 2.3 实验后评估

**决策流程图**:
```
Epoch 10完成
    ↓
获取PSNR
    ↓
对比基线(17.97dB)
    ↓
    ├─> 提升>0.2dB ──> ✅ 保留，继续训练到50epoch
    ├─> 提升0.1-0.2dB ──> ⚠️ 再验证一次
    ├─> 提升<0.1dB ──> ❌ 回滚
    └─> 下降 ──> ❌ 立即回滚
```

**回滚操作**:
```bash
# 方式1: 软回滚（保留commit历史）
git revert HEAD

# 方式2: 硬回滚（删除commit）
git reset --hard HEAD~1
git push --force

# 方式3: 切换回主分支
git checkout main
git branch -D exp/eagle_loss  # 删除实验分支
```

---

## 三、修改分类与规范

### 3.1 损失函数修改

**规范**:
- 单独修改，不与其他超参一起调整
- 必须记录loss权重
- 对比必须包含loss曲线

**模板**:
```python
# code/loss.py

def eagle_loss(pred, target, alpha=0.1):
    """
    EAGLE边缘感知损失
    
    Args:
        alpha: 边缘损失权重 (默认0.1)
    """
    l1 = F.l1_loss(pred, target)
    edge = gradient_loss(pred, target)
    return l1 + alpha * edge
```

**commit message**:
```
feat: 添加EAGLE Loss边缘感知损失 (EXP_20240401_001)

修改:
- code/loss.py: 添加eagle_loss函数

验证:
- Epoch 10 PSNR: 18.45 dB (基线17.97 dB, +0.48dB)
- 满足更优性标准#1 (定量指标)

决策: ✅ 保留，继续训练
```

### 3.2 数据增强修改

**规范**:
- 一次只修改一种增强类型
- 必须记录增强参数
- 对比必须包含训练时间和内存占用

### 3.3 模型架构修改

**规范**:
- 架构修改必须单独commit
- 需要记录参数量和计算量变化
- 必须验证前向传播正常

### 3.4 超参数调整

**规范**:
- 学习率、batch size等单独调整
- 使用grid search时，每个参数单独验证
- 禁止同时调整多个超参数

---

## 四、禁止事项

### 4.1 绝对禁止 ❌

| 禁止行为 | 原因 | 替代方案 |
|----------|------|----------|
| 一次修改>3处 | 无法归因效果 | 分多次commit |
| 无benchmark验证就保留 | 可能引入退化 | 强制验证流程 |
| PSNR下降强行解释 | 自欺欺人 | 果断回滚 |
| 混合多个实验 | 无法区分效果 | 每个实验单独分支 |
| 修改后不记录 | 无法追溯 | 使用实验模板 |

### 4.2 谨慎使用 ⚠️

| 行为 | 风险 | 建议 |
|------|------|------|
| GAN损失 | 假阳性风险高 | 避免使用 |
| 复杂数据增强 | 可能破坏解剖结构 | 可视化检查 |
| 大幅度学习率调整 | 训练不稳定 | 每次调整<2x |
| 模型剪枝 | 细节丢失 | 渐进式微调 |

---

## 五、工具与检查清单

### 5.1 提交前检查清单

```markdown
- [ ] 修改不超过3处
- [ ] 修改在独立分支
- [ ] commit message符合规范
- [ ] 本地测试通过
- [ ] 已记录基线benchmark
- [ ] 实验ID已分配
```

### 5.2 验证后检查清单

```markdown
- [ ] 训练完成5-10epoch
- [ ] 获取最新PSNR/SSIM
- [ ] 与基线公平对比
- [ ] 决策：保留/回滚/再验证
- [ ] 记录决策理由
- [ ] 更新实验记录
```

### 5.3 回滚后检查清单

```markdown
- [ ] 确认回滚到基线commit
- [ ] 验证benchmark恢复基线
- [ ] 记录失败原因
- [ ] 更新实验记录（标记失败）
```

---

## 六、附录

### 6.1 实验记录示例

```markdown
## EXP_20240401_001: EAGLE Loss验证

**修改**: code/loss.py添加eagle_loss (alpha=0.1)

**基线**: commit 6f9f6ed, PSNR=17.97dB

**结果**:
| Epoch | 修改后 | 基线 | 差异 |
|-------|--------|------|------|
| 5 | 18.12 | 17.85 | +0.27 |
| 10 | 18.45 | 17.97 | +0.48 ✅ |

**决策**: 保留，继续训练到50epoch

**备注**: 边缘保持确实更好，视觉上更清晰
```

### 6.2 常用命令

```bash
# 创建实验分支
git checkout -b exp/YYYYMMDD_description

# 查看基线commit
git log --oneline -1 main

# 对比修改
git diff main

# 快速训练验证
python train.py --epoch 10 --net_idx=exp_test

# 获取benchmark
python -c "import json; d=json.load(open('train_log/.../training_history.json')); print(d['val_psnr'][-1])"

# 回滚
git revert HEAD
```

---

*制定: 2026-04-01*  
*生效: 立即*  
*审核: Agent Swarm*
