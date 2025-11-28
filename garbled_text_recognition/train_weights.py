"""
乱码检测权重训练脚本 - 使用逻辑回归
功能：从标注数据学习最优的 ICR/GSR/LRS/PPL/ENT 权重
"""

import json
import numpy as np
import os
import sys

# 获取脚本所在目录，确保路径正确
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# 导入检测器中的指标计算函数
from garbled_text_detector import (
    calculate_icr, calculate_gsr, calculate_lrs, calculate_ppl, calculate_ent,
    detect_primary_language, CONFIG
)


def load_training_data(filepath: str = "training_data.json") -> list:
    """加载训练数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["samples"]


def extract_features(text: str) -> dict:
    """提取文本的五个指标特征"""
    language = detect_primary_language(text)

    icr = calculate_icr(text)
    gsr = calculate_gsr(text)
    lrs = calculate_lrs(text, language)
    ppl = calculate_ppl(text)
    ent = calculate_ent(text)  # 新增：字符熵

    # 归一化
    ppl_norm = min(ppl / CONFIG["THRESHOLDS"]["PPL_MAX"], 1.0)
    ent_norm = min(ent / 5.0, 1.0)  # 最大熵约5.0

    return {
        "ICR": icr,
        "GSR": gsr,
        "LRS": lrs,
        "PPL": ppl,
        "PPL_norm": ppl_norm,
        "ENT": ent,
        "ENT_norm": ent_norm
    }


def prepare_dataset(samples: list) -> tuple:
    """准备训练数据集"""
    print(f"正在提取 {len(samples)} 个样本的特征...")

    X = []  # 特征矩阵
    y = []  # 标签

    for i, sample in enumerate(samples):
        text = sample["text"]
        label = sample["label"]

        features = extract_features(text)

        # 特征向量：将指标转换为"乱码倾向"分数
        # ICR, GSR, PPL 越高越像乱码 (正向)
        # LRS, ENT 越低越像乱码 (反向)
        feature_vector = [
            features["ICR"],           # 无效字符比例
            features["GSR"],           # 乱码符号比例
            1 - features["LRS"],       # 1-可读性 = 不可读性
            features["PPL_norm"],      # 归一化困惑度
            1 - features["ENT_norm"]   # 1-熵 = 低多样性（新增）
        ]

        X.append(feature_vector)
        y.append(label)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(samples)}")

    return np.array(X), np.array(y)


def train_weights(X: np.ndarray, y: np.ndarray) -> dict:
    """
    使用逻辑回归训练权重
    返回：优化后的权重配置
    """
    print("\n开始训练逻辑回归模型...")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练逻辑回归模型
    model = LogisticRegression(
        C=1.0,              # 正则化强度
        class_weight='balanced',  # 处理类别不平衡
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 评估模型
    print("\n=== 模型评估 ===")
    y_pred = model.predict(X_test)
    print(f"测试集准确率: {model.score(X_test, y_test):.2%}")

    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"5折交叉验证准确率: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['正常', '乱码']))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  预测正常  预测乱码")
    print(f"实际正常  {cm[0][0]:^6}  {cm[0][1]:^6}")
    print(f"实际乱码  {cm[1][0]:^6}  {cm[1][1]:^6}")

    # 从模型系数推导权重
    # 逻辑回归的系数表示每个特征对结果的影响程度
    coefficients = model.coef_[0]
    print(f"\n原始系数: {coefficients}")

    # 将系数转换为正权重（取绝对值后归一化）
    abs_coefs = np.abs(coefficients)
    weights = abs_coefs / abs_coefs.sum()

    feature_names = ["ICR", "GSR", "LRS", "PPL", "ENT"]
    weight_dict = {name: round(float(w), 3) for name, w in zip(feature_names, weights)}

    print("\n=== 学习到的权重 ===")
    for name, weight in weight_dict.items():
        print(f"  {name}: {weight:.3f} ({weight*100:.1f}%)")

    # 计算最优阈值
    # 使用训练好的模型在全数据集上预测概率
    proba = model.predict_proba(X_scaled)[:, 1]  # 乱码的概率

    # 计算加权得分（使用新权重）
    weighted_scores = []
    for i in range(len(X)):
        # 计算质量分数（1 - 乱码倾向）
        score = (
            (1 - X[i][0]) * weights[0] +  # 1-ICR
            (1 - X[i][1]) * weights[1] +  # 1-GSR
            (1 - X[i][2]) * weights[2] +  # LRS (已经是1-LRS的反向)
            (1 - X[i][3]) * weights[3] +  # 1-PPL_norm
            (1 - X[i][4]) * weights[4]    # ENT (已经是1-ENT的反向)
        )
        weighted_scores.append(score)

    # 找到最佳分界阈值
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds_roc = roc_curve(y, proba)
    # Youden's J statistic: 最大化 (TPR - FPR)
    j_scores = tpr - fpr
    best_threshold_idx = np.argmax(j_scores)

    # 根据加权分数分布确定 OVERALL_MIN 阈值
    normal_scores = [weighted_scores[i] for i in range(len(y)) if y[i] == 0]
    gibberish_scores = [weighted_scores[i] for i in range(len(y)) if y[i] == 1]

    # 阈值设为两类分数分布的中间值
    suggested_threshold = (min(normal_scores) + max(gibberish_scores)) / 2
    suggested_threshold = round(max(0.3, min(0.7, suggested_threshold)), 2)

    print(f"\n=== 建议阈值 ===")
    print(f"  正常文本得分范围: {min(normal_scores):.3f} ~ {max(normal_scores):.3f}")
    print(f"  乱码文本得分范围: {min(gibberish_scores):.3f} ~ {max(gibberish_scores):.3f}")
    print(f"  建议 OVERALL_MIN: {suggested_threshold}")

    return {
        "weights": weight_dict,
        "suggested_overall_min": suggested_threshold,
        "accuracy": round(model.score(X_test, y_test), 3),
        "cv_accuracy": round(cv_scores.mean(), 3)
    }


def generate_config_code(result: dict) -> str:
    """生成可直接复制到 garbled_text_detector.py 的配置代码"""
    weights = result["weights"]
    threshold = result["suggested_overall_min"]

    code = f'''# 机器学习优化后的权重配置
# 训练准确率: {result["accuracy"]:.1%}, 交叉验证: {result["cv_accuracy"]:.1%}
CONFIG = {{
    "WEIGHTS": {{
        "ICR": {weights["ICR"]},   # 无效字符比例
        "GSR": {weights["GSR"]},   # 乱码符号比例
        "LRS": {weights["LRS"]},   # 语言可读性
        "PPL": {weights["PPL"]},   # 困惑度
        "ENT": {weights["ENT"]}    # 字符熵
    }},

    "THRESHOLDS": {{
        "ICR_MAX": 0.05,
        "GSR_MAX": 0.03,
        "LRS_MIN": 0.55,
        "PPL_MAX": 300,
        "ENT_MIN": 3.5,
        "OVERALL_MIN": {threshold}
    }},

    "PPL_MODEL": "gpt2",
    "MAX_PPL_LENGTH": 1024,
}}'''
    return code


def main():
    print("=" * 60)
    print("乱码检测权重训练 - 逻辑回归")
    print("=" * 60)

    # 1. 加载数据
    samples = load_training_data()
    print(f"\n加载了 {len(samples)} 个训练样本")

    normal_count = sum(1 for s in samples if s["label"] == 0)
    gibberish_count = sum(1 for s in samples if s["label"] == 1)
    print(f"  正常文本: {normal_count} 个")
    print(f"  乱码文本: {gibberish_count} 个")

    if len(samples) < 20:
        print("\n⚠️ 警告: 样本数量较少，建议添加更多样本以提高准确率")

    # 2. 提取特征
    X, y = prepare_dataset(samples)

    # 3. 训练模型
    result = train_weights(X, y)

    # 4. 生成配置代码
    print("\n" + "=" * 60)
    print("生成的配置代码（可直接复制到 garbled_text_detector.py）:")
    print("=" * 60)
    config_code = generate_config_code(result)
    print(config_code)

    # 5. 保存结果
    output_file = "trained_weights.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n训练结果已保存到: {output_file}")

    print("\n" + "=" * 60)
    print("下一步操作:")
    print("1. 将上面的 CONFIG 代码复制到 garbled_text_detector.py")
    print("2. 或运行: uv run python apply_weights.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
