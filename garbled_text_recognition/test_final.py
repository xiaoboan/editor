"""
测试优化后的判定逻辑
"""

import sys
import os
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from garbled_text_detector import detect_gibberish

print("=" * 70)
print("测试优化后的判定逻辑")
print("=" * 70)

test_cases = [
    {
        "name": "正常中文",
        "text": "这是一段正常的中文文本，包含标点符号。应该被检测为有效内容。",
        "expected": False
    },
    {
        "name": "正常英文",
        "text": "This is a normal English text with proper grammar and spelling.",
        "expected": False
    },
    {
        "name": "中英文混合",
        "text": "Hello世界，这是一段中英文混合的text。",
        "expected": False  # 现在应该正确识别为正常
    },
    {
        "name": "真实乱码",
        "text": "����� ���� ���� йййй",
        "expected": True
    },
    {
        "name": "键盘乱输",
        "text": "asdkfj;lkasdf;lkj;lkj asdfasdf",
        "expected": True
    },
    {
        "name": "特殊符号过多",
        "text": "％＆＊（）｛｝［］＃＠！％＆＊（）｛｝",
        "expected": True
    },
    {
        "name": "轻微问题文本",
        "text": "这是正常文本，但有一点小问题###",
        "expected": False  # 只有一个指标可能超标，不应判为乱码
    },
]

print("\n判定规则说明：")
print("-" * 70)
print("1. 需要至少2个指标超标 → 判定为乱码")
print("2. 或综合得分 < 0.60 → 判定为乱码")
print("3. 否则 → 判定为正常文本")
print("-" * 70)

success_count = 0
fail_count = 0

for case in test_cases:
    result = detect_gibberish(case["text"], detailed=True)

    # 判断是否符合预期
    is_correct = result["is_gibberish"] == case["expected"]
    status = "✓" if is_correct else "✗"

    if is_correct:
        success_count += 1
    else:
        fail_count += 1

    print(f"\n{status} {case['name']}:")
    print(f"  文本: {case['text'][:40]}...")
    print(f"  预期: {'乱码' if case['expected'] else '正常'}")
    print(f"  实际: {'乱码' if result['is_gibberish'] else '正常'}")
    print(f"  综合得分: {result['overall_score']:.3f}")
    print(f"  问题数: {len(result.get('issues', []))}")

    if result.get('issues'):
        print(f"  问题: {', '.join(result['issues'])}")

print("\n" + "=" * 70)
print(f"测试结果汇总:")
print(f"  成功: {success_count}/{len(test_cases)}")
print(f"  失败: {fail_count}/{len(test_cases)}")
print(f"  准确率: {success_count/len(test_cases)*100:.1f}%")
print("=" * 70)