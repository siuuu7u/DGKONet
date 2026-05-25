import json
import torch
import numpy as np
import re
# 假设 metrics.py 和 metrics_clinical.py 中的代码已经导入
from metrics import compute_scores
from metrics_clinical import CheXbertMetrics


def clean_report(report, dataset):
    # 清洗 Iu-xray 报告
    if dataset == "iu_xray":
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        report = ' . '.join(tokens) + ' .'
    # 清洗 MIMIC-CXR 报告
    else:
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        report = ' . '.join(tokens) + ' .'
    return report


# 初始化 CheXbert 评估器（根据实际情况调整路径和设备）
checkpoint_path = "/mnt/zzh/models/chexpert/chexbert.pth"
device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
mbatch_size = 16  # 批量处理大小
chexbert_metrics = CheXbertMetrics(checkpoint_path, mbatch_size, device)

# 读取 JSON 文件（你的新格式）
gts = []  # 真实报告（true_report）
res_original = []  # 原始生成报告（original_generated_report）
res_optimized = []  # 优化后报告（optimized_report）

# 配置文件路径
json_filename = "/home/user2/zzhData/Test/models/data/optimized_reports_en_with_mesh_sim_final_590_20251112_020326.json" # 替换为你的JSON文件路径
dataset = "iu_xray"  # 根据实际数据集选择 "iu_xray" 或 "mimic_cxr"

with open(json_filename, 'r', encoding='utf-8') as f:
    data_list = json.load(f)  # 直接加载为列表（JSON根节点是列表）

    for item in data_list:
        # 提取 query_info 中的关键报告
        query_info = item["query_info"]
        true_report = query_info["true_report"]
        original_report = query_info["original_generated_report"]
        # 提取优化后报告
        optimized_report = item.get("optimized_report", original_report)  # 若没有优化报告则用原始报告

        # 清洗报告（根据数据集选择规则）
        true_clean = clean_report(true_report, dataset)
        original_clean = clean_report(original_report, dataset)
        optimized_clean = clean_report(optimized_report, dataset)

        # 存入列表
        gts.append(true_clean)
        res_original.append(original_clean)
        res_optimized.append(optimized_clean)


# 计算通用文本指标（compute_scores）
def calculate_generic_metrics(ground_truths, results):
    # 转换为指标函数要求的格式：{索引: [报告]}
    gts_dict = {i: [gt] for i, gt in enumerate(ground_truths)}
    res_dict = {i: [re] for i, re in enumerate(results)}
    return compute_scores(gts_dict, res_dict)


# 计算原始报告和优化后报告的指标
metrics_original = calculate_generic_metrics(gts, res_original)
metrics_optimized = calculate_generic_metrics(gts, res_optimized)

# 计算医疗专业指标（CheXbert）
chex_original = chexbert_metrics.compute(gts, res_original)
chex_optimized = chexbert_metrics.compute(gts, res_optimized)

# 输出对比结果
print("=" * 50)
print("原始生成报告（original_generated_report）评估结果：")
print("通用文本指标：", metrics_original)
print("医疗专业指标（CheXbert）：", chex_original)
print("\n" + "=" * 50)
print("优化后报告（optimized_report）评估结果：")
print("通用文本指标：", metrics_optimized)
print("医疗专业指标（CheXbert）：", chex_optimized)
print("=" * 50)