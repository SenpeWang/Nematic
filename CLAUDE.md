# CLAUDE.md

## 交流原则

1. **基于事实**：和用户交流的内容不能虚构、夸大，一定基于事实
2. **诚实回答**：
   - 知道就回答"知道"
   - 不知道就回答"不知道"
   - 不确定就回答"不确定"
3. **称呼**：每次回复用户叫用户"Senpe"

## 设计原则

1. **NPR 在每个 block 之前调用**：
   - 更新并演化向列相物理场 Q-tensor
   - 在 StageBackbone 中每个 block 前执行

2. **QTensorTools 是工具函数**：
   - 只做数学运算，不学习参数
   - 所有 Q-tensor 操作都必须通过 QTensorTools

## 深度学习对于模块称呼的定义(不看架构,只看什么定义为层,什么定义为块,什么定义为阶段)

1. **Layer（层）**：最小的可计算单元
   - 基础组件：Conv2d、Linear、GroupNorm、GELU、DropPath
   - 自定义组件：ConvBlock、SS2D、NematicConv、NematicStructureGate

2. **Block（块）**：由多个 layer 组成的模块
   - OFEBlock、OFMBlock、NPR

3. **Stage（阶段）**：由多个 block 组成的阶段
   - Stage 0：NPR + OFEBlock × 2
   - Stage 1：NPR + OFEBlock × 3
   - Stage 2：NPR + OFMBlock × 6
   - Stage 3：NPR + OFMBlock × 2

4. **向列相**：Q-tensor 在 -1,1 范围的定义
   - $q_1, q_2 \in [-1, 1]$
   - $S = \sqrt{q_1^2 + q_2^2} \in [0, 1]$
   - $q_1 = S \cos(2\theta)$
   - $q_2 = S \sin(2\theta)$
   - $\theta = 0.5 \times \text{atan2}(q_2, q_1)$

## 命名规范

1. **GroupNorm**：如果多个 GroupNorm 都是一样的，命名为 GN

## Q-tensor 读写规则

**核心原则：只有 NPR 可以读取和写入（演化更新）向列相信息，其他模块只能读取 NPR 维护的 Q 变量。**

**违反规则的情况：**
- 任何非 NPR 模块修改 Q → 错误
- 任何模块直接修改 Q 的值 → 错误
- 只有 NPR 可以通过低秩预测与软门控凸组合更新 Q：$Q_{new} = Q + \text{gate} \cdot (Q_{pred} - Q)$
