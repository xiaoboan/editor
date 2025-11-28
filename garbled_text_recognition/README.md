# 文本乱码检测API - v1.6

基于GPT-2的文本乱码检测API，通过多维度指标准确判断文本是否包含乱码，支持LLM二次确认。

## 核心功能

- **高准确度检测**：使用GPT-2计算PPL(困惑度)，结合多维指标
- **多语言支持**：支持中文、英文、日语、韩语、德语、俄语自动识别
- **RESTful API**：FastAPI构建，易于集成
- **实时检测**：毫秒级响应
- **机器学习优化**：支持通过训练数据自动优化权重
- **LLM二次确认**：可选的LLM二次校验，提高准确率

## 检测指标详解

### 1. ICR (Invalid Character Ratio) - 无效字符比例
- **原理**：检测文本中的控制字符、替换字符（�）、私有区域Unicode字符等异常字符
- **计算方法**：无效字符数 / 总字符数
- **阈值**：< 5%
- **说明**：正常文本很少包含这类字符，高比例通常表示编码错误或数据损坏

### 2. GSR (Gibberish Symbol Ratio) - 乱码符号比例
- **原理**：检测连续重复的特殊符号或混乱的符号组合
- **计算方法**：识别如 `####`、`@@@@` 或连续5个以上非单词字符
- **阈值**：< 3%
- **说明**：正常文本中符号通常有规律，大量连续特殊符号是乱码特征

### 3. LRS (Language Readability Score) - 语言可读性
- **原理**：基于语言学特征评估文本的可读性
- **计算方法**：
  - **中文**：使用jieba分词，计算有效词汇比例
  - **英文/德语/俄语**：使用拼写检查器，计算正确拼写单词的比例
  - **日语**：使用janome分词，计算实词比例
  - **韩语**：计算韩文字符占比
- **阈值**：> 55%
- **说明**：真实文本应该包含大量有意义的词汇

### 4. PPL (Perplexity) - 困惑度
- **原理**：使用GPT-2语言模型评估文本的语义连贯性
- **计算方法**：基于语言模型计算文本出现的概率，概率越低困惑度越高
- **阈值**：< 300
- **说明**：
  - PPL < 100：非常流畅的文本
  - PPL 100-200：较为自然的文本
  - PPL 200-300：可能有问题但仍可接受
  - PPL > 300：很可能是乱码或无意义文本
- **特点**：这是最智能的指标，能理解语义层面的连贯性

### 5. ENT (Entropy) - 字符熵
- **原理**：衡量文本中字符的多样性，基于香农熵计算
- **计算方法**：`-Σ(p(x) * log2(p(x)))`，其中p(x)是字符出现概率
- **阈值**：> 3.5
- **说明**：
  - ENT > 4.0：正常文本，字符分布均匀
  - ENT 3.5-4.0：边界情况
  - ENT < 3.5：低多样性，可能是重复/简单乱码
- **特点**：专门用于检测单字母重复类乱码（如"nnnmmmnnn"）

## 综合评分机制

```
综合得分 = (1-ICR)×W_ICR + (1-GSR)×W_GSR + LRS×W_LRS + (1-PPL_norm)×W_PPL + ENT_norm×W_ENT
```

- **默认权重**：
  - LRS权重最高（30%），检测无意义文本
  - ICR权重20%，检测编码错误
  - PPL权重18%，评估语义连贯性
  - GSR权重17%，检测符号乱码
  - ENT权重15%，检测低多样性文本
- **判定规则**：
  1. 需要**至少1个指标**严重超标 → 判定为乱码
  2. 或综合得分 < 70% → 判定为乱码
  3. 否则 → 判定为正常文本

### 阈值配置
```python
"THRESHOLDS": {
    "ICR_MAX": 0.05,      # 无效字符阈值
    "GSR_MAX": 0.03,      # 乱码符号阈值
    "LRS_MIN": 0.55,      # 语言可读性阈值
    "PPL_MAX": 300,       # 困惑度阈值
    "ENT_MIN": 3.5,       # 字符熵阈值
    "OVERALL_MIN": 0.70   # 综合得分阈值
}
```

## 快速开始

### 1. 安装依赖

```bash
uv pip install -r requirements.txt
```

### 2. 启动API

```bash
uv run python api.py
```

### 3. 测试API

访问 http://localhost:8000/docs 查看交互式文档

### 4. 调用示例

```python
import requests

# 基础检测
response = requests.post(
    "http://localhost:8000/detect",
    json={
        "text": "这是一段正常的中文文本。",
        "detailed": True
    }
)
result = response.json()
print(f"是否乱码: {result['is_gibberish']}")
print(f"置信度: {result['confidence']}")

# 启用LLM二次确认
response = requests.post(
    "http://localhost:8000/detect",
    json={
        "text": "待检测文本内容...",
        "detailed": True,
        "llm_confirm": True,
        "llm_token": "your-api-token"
    }
)
result = response.json()
print(f"是否乱码: {result['is_gibberish']}")
print(f"LLM确认结果: {result.get('llm_confirmed')}")
```

## API返回示例

```json
{
  "is_gibberish": false,
  "confidence": 0.85,
  "overall_score": 0.78,
  "language": "zh-cn",
  "processing_time_ms": 150.5,
  "metrics": {
    "ICR": {
      "value": 0.0,
      "threshold": 0.05,
      "desc": "无效字符比例"
    },
    "GSR": {
      "value": 0.0,
      "threshold": 0.03,
      "desc": "乱码符号比例"
    },
    "LRS": {
      "value": 0.65,
      "threshold": 0.55,
      "desc": "语言可读性"
    },
    "PPL": {
      "value": 120.5,
      "threshold": 300,
      "desc": "困惑度"
    },
    "ENT": {
      "value": 4.2,
      "threshold": 3.5,
      "desc": "字符熵"
    }
  },
  "llm_confirmed": null
}
```

## LLM二次确认功能

当启用`llm_confirm`参数时，如果初步检测判断为乱码，会调用LLM进行二次确认：

- **触发条件**：仅当初步检测判断为乱码时才调用LLM
- **确认逻辑**：如果LLM判断不是乱码，则以LLM结果为准
- **配置参数**：
  - `llm_confirm`: 是否启用LLM二次确认
  - `llm_token`: LLM API的认证Token

### LLM配置

在`garbled_text_detector.py`中可以修改LLM配置：

```python
LLM_CONFIG = {
    "API_URL": "your-llm-api-url",
    "MODEL": "qwen-flash",
    "TIMEOUT": 30,
    "MAX_TEXT_LENGTH": 10000
}
```

## API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/detect` | POST | 检测文本是否包含乱码 |
| `/health` | GET | 健康检查 |
| `/config` | GET | 获取当前配置 |
| `/` | GET | 服务信息 |

### POST /detect 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 需要检测的文本（1-50000字符） |
| `detailed` | bool | 否 | true | 是否返回详细指标 |
| `llm_confirm` | bool | 否 | false | 是否启用LLM二次确认 |
| `llm_token` | string | 否 | null | LLM API Token |

## 机器学习权重训练

支持使用标注数据自动优化检测权重：

### 1. 准备训练数据

创建`training_data.json`文件：

```json
{
  "samples": [
    {"text": "正常文本示例", "label": 0},
    {"text": "乱码示例文本", "label": 1}
  ]
}
```

### 2. 运行训练

```bash
uv run python train_weights.py
```

训练脚本会：
- 使用逻辑回归学习最优权重
- 输出5折交叉验证准确率
- 生成`trained_weights.json`

### 3. 应用权重

```bash
uv run python apply_weights.py
```

该脚本会自动将训练好的权重更新到`garbled_text_detector.py`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `garbled_text_detector.py` | 核心检测逻辑，包含5个指标计算和LLM确认 |
| `api.py` | FastAPI服务 |
| `train_weights.py` | 机器学习权重训练脚本 |
| `apply_weights.py` | 权重应用脚本 |
| `test_final.py` | 测试脚本 |
| `requirements.txt` | 依赖列表 |
| `training_data.json` | 训练数据（需自行创建） |
| `trained_weights.json` | 训练结果（训练后生成） |

## 依赖库说明

| 库 | 用途 |
|------|------|
| `jieba` | 中文分词 |
| `janome` | 日语分词（纯Python，跨平台） |
| `kiwipiepy` | 韩语分词 |
| `pyspellchecker` | 英文拼写检查 |
| `langdetect` | 语言检测 |
| `transformers` | GPT-2模型加载 |
| `torch` | 深度学习框架 |

## 常见问题

### Q: 为什么正常中文被误判为乱码？
A: 已优化中文处理，降低了LRS阈值，保留中文标点和单字词。如仍有问题，可调整配置中的阈值。

### Q: PPL值很高但文本正常？
A: GPT-2主要训练于英文，对中文的PPL值可能偏高。可以降低PPL权重或提高PPL_MAX阈值。

### Q: 如何调整灵敏度？
A: 修改`garbled_text_detector.py`中的CONFIG配置：
- 降低各阈值 → 更严格
- 提高各阈值 → 更宽松

### Q: LLM二次确认什么时候使用？
A: 当需要更高准确率时启用。注意LLM调用会增加响应时间和API成本。

## 性能说明

- **首次加载**：需要下载GPT-2模型（约500MB）
- **单次检测**：100-500ms（不含LLM确认）
- **LLM确认**：额外增加1-3秒
- **内存占用**：1-2GB（主要是GPT-2模型）
- **并发处理**：支持多请求并发

## 版本历史

### v1.5 (2024-11)
- **新增德语支持**：使用拼写检查（和英语逻辑一致）
- **新增俄语支持**：使用拼写检查（和英语、德语逻辑一致）
- 优化韩语LRS计算：简化为韩文字符占比

### v1.4 (2024-11)
- **新增日语和韩语支持**：
  - 日语：使用janome分词（纯Python，跨平台兼容）
  - 韩语：计算韩文字符占比（kiwipiepy分词复杂计算时间长， 已简化）
- 更新语言检测逻辑，支持平假名、片假名、韩文字符识别
- 更新LRS计算，为日韩语言添加专门的词性分析

### v1.3 (2024-11)
- **新增LLM二次确认**：可选的LLM二次校验功能
- 添加`llm_confirm`和`llm_token`API参数
- 优化检测器代码结构

### v1.2 (2024-11)
- **新增ENT指标**：字符熵检测，用于识别低多样性/重复乱码
- 支持5个检测指标：ICR、GSR、LRS、PPL、ENT
- 调整阈值：PPL_MAX 500→300，LRS_MIN 0.40→0.55
- 更新权重分配，适配新指标
- 优化训练脚本支持5指标

### v1.1 (2024-11)
- **新增机器学习权重**：优化权重配置
- 提高LRS（语言可读性）权重
- 调整阈值以提高检测敏感度

### v1.0 (2024-11)
- 初始版本发布
- 支持中英文乱码检测
- 优化中文处理逻辑
- 调整阈值以减少误判