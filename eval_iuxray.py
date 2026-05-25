import json
import torch
import numpy as np
import re
# 假设 metrics.py 和 metrics_clinical.py 中的代码已经导入
from metrics import compute_scores
from metrics_clinical import CheXbertMetrics

def clean_report(report, dataset):
    # 清洗 Iu - xray 报告clean_report
    if dataset == "iu_xray":
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        report = ' . '.join(tokens) + ' .'
    # 清洗 MIMIC - CXR 报告
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

# 初始化 CheXbert，这里需要根据实际情况填写 checkpoint_path, device
checkpoint_path = "/mnt/zzh/models/chexpert/chexbert.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
mbatch_size = 16  # 定义 mini-batch 大小
chexbert_metrics = CheXbertMetrics(checkpoint_path, mbatch_size, device)

# 读取 output.jsonl 文件
gts = []
res = []
output_filename = "/mnt/zzh/iuxary_output/output4.jsonl"
with open(output_filename, 'r') as f:
    for line in f:
        data = json.loads(line)
        true_report = data["true_report"]
        generated_report = data["generated_report"]
        gts.append(true_report)
        res.append(generated_report)

# 计算 compute_scores 的结果
test_gts_dict = {i: [gt] for i, gt in enumerate(gts)}
test_res_dict = {i: [re] for i, re in enumerate(res)}
test_met = compute_scores(test_gts_dict, test_res_dict)



# 计算 CheXbertMetrics 的结果
# test_ce = chexbert_metrics.compute(gts, res)

# 输出评价指标结果
print("compute_scores 结果:", test_met)
print("CheXbertMetrics 结果:", test_ce)