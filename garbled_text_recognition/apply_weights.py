"""
权重应用脚本 - 将训练好的权重自动更新到 garbled_text_detector.py
"""

import json
import re
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def load_trained_weights(filepath: str = "trained_weights.json") -> dict:
    """加载训练好的权重"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_weights_to_detector(weights_data: dict, detector_file: str = "garbled_text_detector.py"):
    """将权重应用到检测器代码"""

    # 读取原始代码
    with open(detector_file, 'r', encoding='utf-8') as f:
        code = f.read()

    weights = weights_data["weights"]
    threshold = weights_data["suggested_overall_min"]

    # 替换权重配置
    # 匹配 WEIGHTS 字典部分
    weights_pattern = r'("WEIGHTS":\s*\{[^}]+\})'

    new_weights = f'''"WEIGHTS": {{
        "ICR": {weights["ICR"]},   # 无效字符比例 (ML优化)
        "GSR": {weights["GSR"]},   # 乱码符号比例 (ML优化)
        "LRS": {weights["LRS"]},   # 语言可读性 (ML优化)
        "PPL": {weights["PPL"]},   # 困惑度 (ML优化)
        "ENT": {weights["ENT"]}    # 字符熵 (ML优化)
    }}'''

    code = re.sub(weights_pattern, new_weights, code)

    # 替换 OVERALL_MIN 阈值
    threshold_pattern = r'("OVERALL_MIN":\s*)[0-9.]+'
    code = re.sub(threshold_pattern, f'"OVERALL_MIN": {threshold}', code)

    # 写回文件
    with open(detector_file, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f"[OK] 权重已更新到 {detector_file}")
    print(f"\n新权重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
    print(f"\n新阈值:")
    print(f"  OVERALL_MIN: {threshold}")


def main():
    print("=" * 50)
    print("应用训练好的权重")
    print("=" * 50)

    try:
        weights_data = load_trained_weights()
        print(f"\n训练准确率: {weights_data['accuracy']:.1%}")
        print(f"交叉验证准确率: {weights_data['cv_accuracy']:.1%}")

        apply_weights_to_detector(weights_data)

        print("\n" + "=" * 50)
        print("完成! 现在可以重新运行 API 测试验证效果")
        print("=" * 50)

    except FileNotFoundError:
        print("[ERROR] 找不到 trained_weights.json")
        print("请先运行 train_weights.py 进行训练")


if __name__ == "__main__":
    main()
