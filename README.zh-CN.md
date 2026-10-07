# 对比模型架构发布包

本发布版本整理于 **2026-10-08**，提供七个对比模型架构及
[补充材料PDF](Supplementary%20Material.pdf)。PDF按用户提供的原文件上传，
未修改内容。本仓库不包含PTCN主模型或其训练流程。
旧Appendix.pdf及README已从当前文件树替换，原内容保留在Git历史中。

版本：**0.2.0，2026-10-05**。完整架构细节参见 [英文README](README.md)
及[修订架构表](ARCHITECTURE_TABLE.md)。

## 范围

本文件夹可以独立作为GitHub仓库根目录。仅包含七种对比模型的架构代码、
固定版本依赖、架构验证测试及文档。不包含数据集、训练权重、训练循环、
诊断流程、实验日志或性能排名，也不依赖原项目目录。

## 模型对应关系

所有模型文件集中在 `comparison_models/`：

| 表中名称 | 文件 | 统一接口名称 | 说明 |
|---|---|---|---|
| TCN | `tcn.py` | `tcn` | 宽核CNN加四个TCN残差块，不是单独TCN |
| CNN | `cnn.py` | `cnn` | 三层64通道Classic CNN，不是155K参数匹配版 |
| DCMS | `dcms.py` | `dcms` | 双通道多尺度1D CNN |
| RMFF | `rmff.py` | `rmff` | 经用户确认，保留3阶段，每阶段2/3/2个模块，共7个 |
| LSTM | `lstm.py` | `lstm` | 128及64单元的双层LSTM |
| V-BiLSTM | `v_bilstm.py` | `v_bilstm` | 恢复历史中值滤波、Wavelet/KCFP、Attention、BiLSTM版本 |
| ROAT | `roat.py` | `roat` | 内置ROAT谱前端和小型CNN，直接输入时域信号 |

**版本纠正：v0.2.0中的V-BiLSTM是真正的历史BiLSTM版本。** 4头Attention
（key_dim=8）后接LayerNorm，再接每方向150单元的BiLSTM，拼接成300维后
经过Dropout(0.5)和七分类Dense。总参数233,227，不包含旧CNN分类头。
已通过历史备份源码与已保存模型核查，但未重新评估准确率或FLOPs。

“完全相同”的核查范围是表中明确列出的结构设置；表中省略的分支宽度、
偏置、BN默认值和自定义层运算沿用历史代码，不另行猜测。RMFF的3阶段解释
已由用户确认。V-BiLSTM模块直接恢复历史备份，含中值滤波；其余六个架构
文件与v0.1.0逐字节相同。旧发布文件夹和ZIP保留，不覆盖。

## 使用

在本文件夹中新建虚拟环境后运行：

```bash
python -m pip install .
python -m comparison_models --model all
python -m comparison_models --model v_bilstm --summary
python -m unittest discover -s tests -v
```

```python
import tensorflow as tf
from comparison_models import build_model

model = build_model("cnn")
y = model(tf.zeros((1, 256, 1), dtype=tf.float32), training=False)
```

统一接口固定输入 `(256, 1)`、七分类，batch维度可变，采用float32默认策略。
返回随机初始化、未编译模型。直接调用底层构建函数修改超参数后，不再属于
本表默认架构。自定义层重载请使用英文README中的 `CUSTOM_OBJECTS` 示例。

本地验证环境为Windows CPU、Python 3.8.20、TensorFlow 2.13.0、Keras 2.13.1、
NumPy 1.24.3。未验证Keras 3；GitHub CI工作流尚未在远端执行。

## 表中训练参数

Adam、学习率0.001、分类交叉熵、batch size 32、1 epoch，统一记录在
`table_protocol.json`，仅作为元数据。本包不提供训练代码，不推断数据划分、
归一化、增强、种子或迭代次数，不声称复现任何历史性能。

## 可追溯性与发布前事项

- `source_manifest.json`：原文件相对路径、SHA-256及提取定义的AST哈希。
- `tests/test_architectures.py`：无需数据的结构、前向与序列化验证。
- `VALIDATION.md`：本次实际验证结果与限制。
- `NOTICE.md`：保留的文献归属说明；这些是本地适配实现，不是作者官方代码。
- 上传前需由代码所有者确认授权并选择许可证，当前没有代为添加开源授权。

原项目、训练权重和历史记录均保持不变。本次整理不是重新训练，也不是
关于模型复杂度或分类性能的新实验。
