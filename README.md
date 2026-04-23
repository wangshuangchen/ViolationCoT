# ViolationCoT: Power Safety Violation Detection with Chain-of-Thought Reasoning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-Latest-green.svg)](https://github.com/hiyouga/LLaMA-Factory)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📖 项目简介

ViolationCoT 是一个基于多模态大语言模型（MLLM）的电力安全违规检测系统。本项目利用 **LLaMA-Factory** 框架，通过思维链（Chain-of-Thought, CoT）推理方法，实现对电力场景中违规行为的智能识别、定位和分析。

### 核心特性

- 🔍 **精准违规检测**: 支持多种电力场景下的违规行为识别
- 🎯 **坐标定位**: 输出违规区域的精确边界框坐标 `[[x1,y1],[x2,y2]]`
- 💭 **思维链推理**: 提供详细的分析过程和推理逻辑
- 🔄 **多模型支持**: 支持 Qwen-VL、InternVL、LLaVA、Gemma、Llama 等多种视觉语言模型
- ⚡ **LoRA 微调**: 高效的参数微调策略，降低训练成本
- 📊 **全面评估**: 内置 IoU、GIoU、DIoU、CIoU 等多种评估指标

## 🏗️ 项目结构

```
ViolationCoT/
├── dataset/                    # 数据集目录
│   └── train-6000.json        # 训练数据集（6000条样本）
├── evaluation/                 # 评估脚本目录
│   ├── eval_qwen3-vl-2B.py           # Qwen3-VL-2B 基础模型评估
│   ├── eval_qwen3-vl-2B-lora.py      # Qwen3-VL-2B LoRA 微调评估
│   ├── eval_qwen2.5-vl-3B.py         # Qwen2.5-VL-3B 基础模型评估
│   ├── eval_qwen2.5-vl-3B-lora.py    # Qwen2.5-VL-3B LoRA 微调评估
│   ├── eval_InternVL3_5-2B.py        # InternVL3.5-2B 评估
│   ├── eval_InternVL3_5-8B.py        # InternVL3.5-8B 评估
│   ├── eval_InternVL3_5-14B.py       # InternVL3.5-14B 评估
│   ├── eval_llava-1.5-7b.py          # LLaVA-1.5-7B 评估
│   ├── eval_llava-1.5-13b.py         # LLaVA-1.5-13B 评估
│   ├── eval_llama3.2_11b.py          # Llama-3.2-11B 评估
│   ├── eval_gemma-3-12b-pt.py        # Gemma-3-12B 评估
│   └── evaluate_results.py           # 结果综合评估脚本
├── train_lora/                 # LoRA 训练配置文件
│   ├── Qwen3-VL.yaml          # Qwen3-VL 训练配置
│   ├── Qwen2.5.yaml           # Qwen2.5-VL 训练配置
│   ├── InternVL3_5.yaml       # InternVL3.5 训练配置
│   ├── llava-1.5.yaml         # LLaVA-1.5 训练配置
│   ├── Llama-3.2.yaml         # Llama-3.2 训练配置
│   └── gemma3-pt.yaml         # Gemma-3 训练配置
└── README.md                  # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (推荐使用 GPU 进行训练和推理)
- LLaMA-Factory 框架

### 安装步骤

1. **克隆 LLaMA-Factory 仓库**

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

2. **安装依赖**

```bash
pip install -e ".[torch,metrics]"
```

3. **安装额外依赖**

```bash
pip install peft transformers accelerate datasets Pillow jsonlines qwen-vl-utils
```

4. **准备数据集**

将您的数据集放置在 `LLaMA-Factory/data/` 目录下，格式如下：

```json
{
  "image_path": "path/to/image.jpg",
  "coordinates": [[x1, y1], [x2, y2]],
  "type": "violation_type",
  "conversation": {
    "Question": "问题描述...",
    "Answer": "标准答案..."
  }
}
```

5. **注册数据集**

在 `LLaMA-Factory/data/dataset_info.json` 中添加您的数据集配置：

```json
{
  "12346-6000": {
    "file_name": "train-6000.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

## 🎓 模型训练

### 支持的模型

| 模型 | 参数量 | 配置文件 |
|------|--------|----------|
| Qwen3-VL | 2B | `Qwen3-VL.yaml` |
| Qwen2.5-VL | 3B | `Qwen2.5.yaml` |
| InternVL3.5 | 2B/8B/14B | `InternVL3_5.yaml` |
| LLaVA-1.5 | 7B/13B | `llava-1.5.yaml` |
| Llama-3.2 | 11B | `Llama-3.2.yaml` |
| Gemma-3 | 12B | `gemma3-pt.yaml` |

### 训练流程

#### 1. 修改训练配置文件

以 Qwen3-VL 为例，编辑 `train_lora/Qwen3-VL.yaml`：

```yaml
### model
model_name_or_path: /path/to/Qwen3-VL-2B-Instruct
image_max_pixels: 4000000
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: 12346-6000
template: qwen2_vl  # 根据模型选择合适的 template
cutoff_len: 1024
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Qwen3-VL-2B-Instruct-12346-6000/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true  # 如果支持 bf16，建议开启
ddp_timeout: 180000000

### eval
eval_dataset: eval-1000-five
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 50
```

#### 2. 启动训练

使用 LLaMA-Factory 命令行工具启动训练：

```bash
llamafactory-cli train train_lora/Qwen3-VL.yaml
```

或使用多 GPU 分布式训练：

```bash
FORCE_TORCHRUN=1 llamafactory-cli train train_lora/Qwen3-VL.yaml
```

#### 3. 监控训练过程

- 查看训练日志：`saves/Qwen3-VL-2B-Instruct-12346-6000/lora/sft/trainer_log.txt`
- 可视化损失曲线：`saves/Qwen3-VL-2B-Instruct-12346-6000/lora/sft/loss.png`

## 📊 模型评估

### 评估脚本说明

每个模型都有对应的评估脚本（基础版和 LoRA 微调版）：

- `eval_xxx.py`: 评估基础模型
- `eval_xxx-lora.py`: 评估 LoRA 微调后的模型

### 运行评估

#### 1. 修改评估脚本配置

以 `eval_qwen3-vl-2B-lora.py` 为例：

```python
# 修改模型路径
model_path = "/path/to/Qwen3-VL-2B-Instruct"

# 修改 LoRA 权重路径
model = PeftModel.from_pretrained(
    model, 
    "/path/to/saves/Qwen3-VL-2B-Instruct-12346-6000/lora/sft"
)

# 修改数据路径
input_file = "/path/to/test-1642.jsonl"
output_file = "/path/to/results/Qwen3-VL-2B-lora-test-1642.jsonl"
image_dir = "/path/to/test-images-1642"
```

#### 2. 执行评估

```bash
python evaluation/eval_qwen3-vl-2B-lora.py
```
#### 3. 综合评估结果

使用 `evaluate_results.py` 计算各项指标：

```bash
python evaluation/evaluate_results.py
```

该脚本会计算：
- **IoU** (Intersection over Union): 交并比
- **GIoU** (Generalized IoU): 广义交并比
- **DIoU** (Distance IoU): 距离交并比
- **CIoU** (Complete IoU): 完整交并比
- **准确率**: 分类准确率
- **提取错误统计**: 坐标提取失败案例分析

评估结果将保存为 TXT 格式，包含详细的统计信息和错误分析。

## 📈 实验结果

详细的实验结果请参考 `../实验结果/实验结果统计表.md` 文件。

### 主要发现

- LoRA 微调显著提升模型在电力违规检测任务上的性能
- 不同规模的模型在不同指标上各有优势
- 思维链推理有效提升了模型的可解释性

## 🔧 常见问题

### Q1: 如何选择合适的 batch_size？

A: 根据您的 GPU 显存大小调整：
- 24GB 显存：建议使用 batch_size=8-16
- 40GB+ 显存：可使用 batch_size=16-32
- 如遇 OOM 错误，请减小 batch_size 或启用梯度累积

### Q2: 训练时出现显存不足怎么办？

A: 可以尝试以下方法：
1. 减小 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 开启 `bf16: true`
4. 减小 `cutoff_len`
5. 使用 DeepSpeed ZeRO-3（取消注释 `deepspeed` 配置）

### Q3: 如何添加新的模型支持？

A: 步骤如下：
1. 在 `train_lora/` 中创建新的 YAML 配置文件
2. 参考现有评估脚本，创建对应的评估文件
3. 确保模板（template）与模型匹配
4. 在 `dataset_info.json` 中注册新数据集

### Q4: 评估结果中的坐标格式是什么？

A: 坐标格式为归一化坐标 `[[x1, y1], [x2, y2]]`，其中：
- `(x1, y1)`: 左上角坐标
- `(x2, y2)`: 右下角坐标
- 坐标值范围：[0, 1]

## 📝 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{violationcot2024,
  title={ViolationCoT: A Power Safety Violation Detection Framework with Chain-of-Thought Reasoning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<div align="center">

**⭐ 如果本项目对您有帮助，请给我们一个 Star！**

Made with ❤️ by ViolationCoT Team

</div>
