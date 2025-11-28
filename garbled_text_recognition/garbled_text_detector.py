"""
文本乱码检测API - 简洁版
主要功能：接收文本，检测是否包含乱码
优先准确度，使用GPT-2计算PPL指标
"""

import re
import math
import json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from spellchecker import SpellChecker
from langdetect import detect_langs
import jieba
from janome.tokenizer import Tokenizer as JanomeTokenizer  # 日语分词
from collections import Counter
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# 全局配置
CONFIG = {
    # 权重配置（优化后）
    "WEIGHTS": {
        "ICR": 0.20,  # 无效字符比例
        "GSR": 0.17,  # 乱码符号比例
        "LRS": 0.30,  # 语言可读性
        "PPL": 0.18,  # 困惑度
        "ENT": 0.15,  # 字符熵（新增：检测低多样性文本）
    },
    # 阈值配置
    "THRESHOLDS": {
        "ICR_MAX": 0.05,  # >5%无效字符 = 高风险
        "GSR_MAX": 0.03,  # >3%乱码符号 = 高风险
        "LRS_MIN": 0.55,  # <55%有效词 = 高风险
        "PPL_MAX": 300,  # >300困惑度 = 高风险
        "ENT_MIN": 3.5,  # <3.5字符熵 = 高风险（正常文本通常>4.0）
        "OVERALL_MIN": 0.7,  # 综合得分<70% = 乱码
    },
    # 语言模型配置
    "PPL_MODEL": "gpt2",  # 使用标准GPT-2保证准确度
    "MAX_PPL_LENGTH": 1024,  # 计算PPL的最大文本长度
}

# 全局模型缓存
MODEL_CACHE = {
    "ppl_model": None,
    "ppl_tokenizer": None,
    "spell_checkers": {},  # 统一管理所有语言的拼写检查器
    "janome": None,
}


def load_ppl_model():
    """加载PPL计算模型（单例模式）"""
    if MODEL_CACHE["ppl_model"] is None:
        print("正在加载GPT-2模型...")
        model_name = CONFIG["PPL_MODEL"]
        MODEL_CACHE["ppl_tokenizer"] = AutoTokenizer.from_pretrained(model_name)
        MODEL_CACHE["ppl_model"] = AutoModelForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_CACHE["ppl_model"].to(device)
        MODEL_CACHE["ppl_model"].eval()
        print(f"GPT-2模型已加载到 {device}")
    return MODEL_CACHE["ppl_model"], MODEL_CACHE["ppl_tokenizer"]


def calculate_ppl(text: str) -> float:
    """
    计算文本困惑度(PPL)
    返回值越低表示文本越自然流畅
    """
    if not text.strip():
        return float("inf")

    try:
        model, tokenizer = load_ppl_model()
        device = model.device

        # 截断过长的文本
        if len(text) > CONFIG["MAX_PPL_LENGTH"]:
            text = text[: CONFIG["MAX_PPL_LENGTH"]]

        # 添加padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        encodings = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )
        input_ids = encodings.input_ids.to(device)

        # 如果文本太短，返回默认值
        if input_ids.size(1) < 2:
            return 100.0

        max_length = model.config.n_positions
        stride = 512
        seq_len = input_ids.size(1)

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / seq_len).item()
        return min(ppl, 10000)  # 设置上限避免极端值
    except Exception as e:
        print(f"PPL计算错误: {e}")
        return 500.0  # 返回默认中等困惑度


def calculate_icr(text: str) -> float:
    """
    计算无效字符比例(ICR)
    检测控制字符、替换字符等
    """
    if not text:
        return 1.0

    # 定义无效字符：控制字符、替换字符(U+FFFD)、私有区域字符
    invalid_pattern = r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F\uFFFD]|[\uE000-\uF8FF]"
    invalid_chars = re.findall(invalid_pattern, text)
    return len(invalid_chars) / len(text)


def calculate_gsr(text: str) -> float:
    """
    计算乱码符号比例(GSR)
    检测连续重复的异常符号
    """
    if not text:
        return 1.0

    # 检测连续的非常规字符
    # 1. 连续的特殊符号（排除常见标点）
    gibberish_pattern1 = r'([^\w\s.,;:!?\'\"()（）【】《》，。！？；：""' "\\-])\1{2,}"
    # 2. 混乱的符号组合
    gibberish_pattern2 = r"[^\w\s]{5,}"  # 连续5个以上非单词字符

    # 使用 finditer 获取完整匹配，findall 在有捕获组时只返回捕获组内容
    matches1 = [m.group(0) for m in re.finditer(gibberish_pattern1, text)]
    matches2 = re.findall(gibberish_pattern2, text)

    gibberish_chars = sum(len(match) for match in matches1) + sum(
        len(match) for match in matches2
    )
    return min(gibberish_chars / len(text), 1.0)


def calculate_ent(text: str) -> float:
    """
    计算字符熵(ENT)
    衡量文本中字符的多样性，低熵表示重复模式
    正常文本熵值通常在4.0-5.0之间
    """
    if not text or len(text) < 2:
        return 0.0

    # 移除空白字符后计算
    clean_text = re.sub(r"\s+", "", text.lower())
    if not clean_text:
        return 0.0

    # 计算字符频率
    char_counts = Counter(clean_text)
    total_chars = len(clean_text)

    # 计算香农熵
    entropy = 0.0
    for count in char_counts.values():
        if count > 0:
            prob = count / total_chars
            entropy -= prob * math.log2(prob)

    return entropy


def detect_primary_language(text: str) -> str:
    """检测文本主要语言，支持中文、英文、日语、韩语"""
    if not text.strip():
        return "unknown"

    try:
        # 检查各语言字符集
        # 日语：平假名 + 片假名
        has_japanese = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))
        # 韩语：韩文字母
        has_korean = bool(re.search(r"[\uac00-\ud7af\u1100-\u11ff]", text))
        # 中文：CJK统一汉字（注意日语也使用汉字，需要结合假名判断）
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        # 英文
        has_english = bool(re.search(r"[a-zA-Z]", text))

        # 判定逻辑（按特征明确性优先）
        # 1. 有假名 -> 日语
        if has_japanese:
            return "ja"
        # 2. 有韩文 -> 韩语
        if has_korean:
            return "ko"
        # 3. 有中文字符且无假名无韩文 -> 中文
        if has_chinese and not has_english:
            return "zh-cn"
        # 4. 纯英文
        if has_english and not has_chinese:
            # 使用langdetect进一步确认
            clean_sample = re.sub(r"[^\w\s]", "", text[:500])
            if clean_sample.strip():
                try:
                    langs = detect_langs(clean_sample)
                    if langs and langs[0].prob > 0.3:
                        return langs[0].lang
                except:
                    pass
            return "en"
        # 5. 中英混合：按字符比例判断
        if has_chinese and has_english:
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            if chinese_chars > english_chars:
                return "zh-cn"
            else:
                return "en"

        # 其他情况，尝试langdetect
        clean_sample = re.sub(r"[^\w\s]", "", text[:500])
        if clean_sample.strip():
            langs = detect_langs(clean_sample)
            if langs and langs[0].prob > 0.3:
                return langs[0].lang

        return "unknown"
    except:
        # 异常时的后备方案
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):
            return "ja"
        if re.search(r"[\uac00-\ud7af]", text):
            return "ko"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh-cn"
        if re.search(r"[a-zA-Z]", text):
            return "en"
        return "unknown"


def _spell_check_lrs(text: str, lang: str = "en") -> float:
    """
    通用拼写检查LRS计算（适用于英语/德语/俄语等拉丁/西里尔字母语言）
    """
    clean_text = re.sub(r"[^\w\s]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return 0.0

    words = clean_text.split()
    if not words:
        return 0.0

    # 获取或创建对应语言的拼写检查器
    if lang not in MODEL_CACHE["spell_checkers"]:
        MODEL_CACHE["spell_checkers"][lang] = SpellChecker(language=lang)

    spell = MODEL_CACHE["spell_checkers"][lang]
    words_lower = [w.lower() for w in words if w.isalpha()]

    if not words_lower:
        return 0.0

    misspelled = spell.unknown(words_lower)
    return 1.0 - (len(misspelled) / len(words_lower))


def calculate_lrs(text: str, language: str) -> float:
    """
    计算语言可读性指标(LRS)
    基于有效词汇的比例，支持中文、英文、日语、韩语
    """
    if not text or not text.strip():
        return 0.0

    if language == "ja":
        # 日语处理：使用janome分词
        try:
            # 初始化janome分词器（单例模式）
            if MODEL_CACHE["janome"] is None:
                MODEL_CACHE["janome"] = JanomeTokenizer()

            tokenizer = MODEL_CACHE["janome"]

            # 清理文本，保留日语字符（假名、汉字）和标点
            clean_text = re.sub(
                r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s。、！？「」『』（）]",
                " ",
                text,
            )
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            if not clean_text:
                return 0.0

            # janome分词
            tokens = list(tokenizer.tokenize(clean_text))

            if not tokens:
                return 0.0

            # 有效词性：名词、动词、形容词、副词等实词
            valid_pos_prefixes = ("名詞", "動詞", "形容詞", "副詞", "接続詞", "感動詞")
            valid_words = [
                t.surface
                for t in tokens
                if any(
                    t.part_of_speech.startswith(prefix) for prefix in valid_pos_prefixes
                )
                or len(t.surface) > 1
            ]

            return len(valid_words) / len(tokens) if tokens else 0.0
        except Exception as e:
            print(f"日语LRS计算错误: {e}")
            return 0.5  # 返回中等值

    elif language == "ko":
        # 韩语处理：直接计算韩文字符占比（简洁有效）
        # 移除空白和常见标点
        clean_text = re.sub(
            r'[\s.,;:!?\'\"()\[\]{}<>@#$%^&*\-_+=|\\~/，。！？；：""' "（）【】《》、]",
            "",
            text,
        )
        if not clean_text:
            return 0.0

        # 韩文字符数量（韩文音节范围）
        korean_chars = len(re.findall(r"[\uac00-\ud7af]", clean_text))
        total_chars = len(clean_text)

        return korean_chars / total_chars if total_chars else 0.0

    elif language in ("de", "ru"):
        # 德语/俄语处理：拼写检查
        return _spell_check_lrs(text, language)

    elif language.startswith("zh"):
        # 中文处理：使用jieba分词
        # 保留中文文本的完整性，不过度清理
        # 移除明显的乱码字符但保留中文标点
        clean_text = re.sub(
            r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\w\s，。！？；：""'
            "（）【】《》、]",
            " ",
            text,
        )
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return 0.0

        words = jieba.lcut(clean_text)

        # 过滤掉纯空白和纯数字
        words = [w for w in words if w.strip() and not w.isdigit()]

        if not words:
            return 0.0

        # 统计有效词：
        # - 长度大于1的词
        # - 或包含中文字符的单字（避免过滤掉单字中文词）
        valid_words = [
            w for w in words if len(w) > 1 or re.search(r"[\u4e00-\u9fff]", w)
        ]

        # 计算有效词比例
        return len(valid_words) / len(words) if words else 0.0
    else:
        # 英文及其他语言：拼写检查
        return _spell_check_lrs(text, "en")


def detect_gibberish(
    text: str, detailed: bool = True, llm_confirm: bool = False, llm_token: str = None
) -> Dict:
    """
    检测文本是否包含乱码

    Args:
        text: 需要检测的文本
        detailed: 是否返回详细指标
        llm_confirm: 是否启用LLM二次确认（当判断为乱码时）
        llm_token: LLM API Token（启用llm_confirm时需要）

    Returns:
        包含检测结果的字典
    """
    if not text or not text.strip():
        return {
            "is_gibberish": True,
            "confidence": 1.0,
            "reason": "空文本或纯空白字符",
            "overall_score": 0.0,
        }

    # 1. 检测语言
    language = detect_primary_language(text)

    # 2. 计算各项指标
    icr = calculate_icr(text)
    gsr = calculate_gsr(text)
    lrs = calculate_lrs(text, language)
    ppl = calculate_ppl(text)
    ent = calculate_ent(text)  # 新增：字符熵

    # 3. 归一化指标 (将指标映射到0-1范围)
    ppl_norm = min(ppl / CONFIG["THRESHOLDS"]["PPL_MAX"], 1.0)
    # 熵归一化：假设最大熵约为5.0（26个字母均匀分布时约4.7）
    ent_norm = min(ent / 5.0, 1.0)

    # 4. 计算综合得分 (0-1, 越高越好)
    weights = CONFIG["WEIGHTS"]
    overall_score = (
        (1 - icr) * weights["ICR"]
        + (1 - gsr) * weights["GSR"]
        + lrs * weights["LRS"]
        + (1 - ppl_norm) * weights["PPL"]
        + ent_norm * weights["ENT"]  # 熵越高越好
    )

    # 5. 判定是否为乱码
    thresholds = CONFIG["THRESHOLDS"]
    breached_thresholds = []

    if icr > thresholds["ICR_MAX"]:
        breached_thresholds.append(f"无效字符过多({icr:.2%})")
    if gsr > thresholds["GSR_MAX"]:
        breached_thresholds.append(f"乱码符号过多({gsr:.2%})")
    if lrs < thresholds["LRS_MIN"]:
        breached_thresholds.append(f"可读性过低({lrs:.2%})")
    if ppl > thresholds["PPL_MAX"]:
        breached_thresholds.append(f"困惑度过高({ppl:.1f})")
    if ent < thresholds["ENT_MIN"]:
        breached_thresholds.append(f"字符熵过低({ent:.2f})")

    # 判定规则：需要至少2个指标超标，或综合得分过低
    # 这样避免单个指标（如混合文本的LRS）导致误判
    is_gibberish = (
        len(breached_thresholds) >= 1 or overall_score < thresholds["OVERALL_MIN"]
    )

    # 计算置信度
    if is_gibberish:
        # 超标指标越多，置信度越高
        confidence = min(0.7 + len(breached_thresholds) * 0.1, 1.0)
    else:
        # 得分越高，置信度越高
        confidence = min(overall_score + 0.2, 1.0)

    result = {
        "is_gibberish": is_gibberish,
        "confidence": round(confidence, 3),
        "overall_score": round(overall_score, 3),
        "language": language,
    }

    if detailed:
        result["metrics"] = {
            "ICR": {
                "value": round(icr, 4),
                "threshold": thresholds["ICR_MAX"],
                "desc": "无效字符比例",
            },
            "GSR": {
                "value": round(gsr, 4),
                "threshold": thresholds["GSR_MAX"],
                "desc": "乱码符号比例",
            },
            "LRS": {
                "value": round(lrs, 4),
                "threshold": thresholds["LRS_MIN"],
                "desc": "语言可读性",
            },
            "PPL": {
                "value": round(ppl, 2),
                "threshold": thresholds["PPL_MAX"],
                "desc": "困惑度",
            },
            "ENT": {
                "value": round(ent, 2),
                "threshold": thresholds["ENT_MIN"],
                "desc": "字符熵",
            },
        }
        if breached_thresholds:
            result["issues"] = breached_thresholds

    # LLM二次确认（仅当启用且初步判断为乱码时）
    if llm_confirm and is_gibberish:
        llm_result = llm_confirm_gibberish(text, llm_token)
        result["llm_confirmed"] = llm_result
        # 如果LLM判断不是乱码，则以LLM结果为准
        if llm_result is False:
            result["is_gibberish"] = False
            result["confidence"] = 0.6  # 降低置信度，表示有分歧

    return result


# LLM配置
LLM_CONFIG = {
    "API_URL": "http://rd-gateway.patsnap.io/compute/openai_chatgpt_turbo",
    "MODEL": "qwen-flash",
    "TIMEOUT": 30,
    "MAX_TEXT_LENGTH": 10000,
}


def llm_confirm_gibberish(text: str, token: str) -> Optional[bool]:
    """
    使用LLM二次确认文本是否为乱码

    Args:
        text: 需要确认的文本
        token: LLM API Token

    Returns:
        True: LLM确认是乱码
        False: LLM认为不是乱码
        None: 调用失败或无法判断
    """
    if not token:
        print("[WARN] LLM token未提供，跳过二次确认")
        return None

    # 截断过长文本
    if len(text) > LLM_CONFIG["MAX_TEXT_LENGTH"]:
        text = text[: LLM_CONFIG["MAX_TEXT_LENGTH"]]

    prompt = f"""### 背景
下面是一段专利说明书文本，该文本是从专利PDF OCR的，请你判断是否有乱码影响阅读，是否需要重新OCR。

### 要求
1、主要是检测文本是否有大段明显乱码，允许少量错误，OCR允许部分O错，整体80%可阅读不视为乱码
2、文本是截取的前1000个字符，因此段落可能不完整，不视为乱码
4、可能会有一些html标签，这些不算做乱码，例如：<b class="d_n">[0001]</b>（段落号标签）、<img src='' class="img-anchor img-center" img-id="IMGF000004_0001"/>（图片标签）

### 返回格式
直接返回json，例如：
{{"is_gibberish":true}}

### 专利文本：
{text}"""

    try:
        response = requests.post(
            LLM_CONFIG["API_URL"],
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
            json={"message": prompt, "model": LLM_CONFIG["MODEL"]},
            timeout=LLM_CONFIG["TIMEOUT"],
        )

        if response.status_code == 200:
            data = response.json()
            # 尝试解析返回的内容
            content = data["data"]["message"]

            # 提取JSON部分
            json_match = re.search(r"\{[^}]+\}", content)
            if json_match:
                llm_result = json.loads(json_match.group())
                return llm_result.get("is_gibberish", None)

            # 如果无法解析JSON，尝试简单判断
            if "true" in content.lower():
                return True
            elif "false" in content.lower():
                return False

        print(f"[WARN] LLM调用失败: {response.status_code}")
        return None

    except Exception as e:
        print(f"[WARN] LLM调用异常: {e}")
        return None


# 预加载模型（可选，提升首次响应速度）
def preload_models():
    """预加载所有模型"""
    print("预加载模型中...")
    load_ppl_model()
    MODEL_CACHE["spell_checkers"]["en"] = SpellChecker(language="en")
    print("模型预加载完成")


if __name__ == "__main__":
    # 测试示例
    test_texts = [
        "This is a normal English text that should be detected as valid.",
        "这是一段正常的中文文本，应该被检测为有效文本。",
        "これは正常な日本語のテキストです。検出されるべきです。",  # 日语正常文本
        "이것은 정상적인 한국어 텍스트입니다.",  # 韩语正常文本
        "Dies ist ein normaler deutscher Text, der erkannt werden sollte.",  # 德语正常文本
        "Это нормальный русский текст, который должен быть обнаружен.",  # 俄语正常文本
        "�����вв��� ���� ����",  # 典型乱码
        "asdkfj;lkasdf;lkj;lkj;lkj",  # 无意义字符
        "％＆＊（）｛｝［］＃＠！",  # 特殊符号
    ]

    print("=" * 50)
    print("文本乱码检测测试（支持中英日韩德俄）")
    print("=" * 50)

    for i, text in enumerate(test_texts, 1):
        print(f"\n测试 {i}:")
        print(f"文本: {text[:50]}..." if len(text) > 50 else f"文本: {text}")
        result = detect_gibberish(text)
        print(f"语言: {result['language']}")
        print(f"检测结果: {'乱码' if result['is_gibberish'] else '正常'}")
        print(f"置信度: {result['confidence']:.1%}")
        print(f"综合得分: {result['overall_score']:.3f}")
        if "issues" in result:
            print(f"问题: {', '.join(result['issues'])}")
