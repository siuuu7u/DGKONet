import os
import json
import re
import torch
from swift.llm import (
    PtEngine, RequestConfig, BaseArguments, InferRequest, safe_snapshot_download
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def clean_report(report, dataset):
    if dataset == "iu_xray":
        report_cleaner = lambda t: t.replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        report = ' . '.join(tokens) + ' .'
    return report

def infer(engine, infer_request):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    gen_list = engine.infer([infer_request], request_config)
    return gen_list[0].choices[0].message.content

def load_model(model_id_or_path):
    adapter_path = safe_snapshot_download(model_id_or_path)
    args = BaseArguments.from_pretrained(adapter_path)
    return PtEngine(args.model, adapters=[adapter_path])

def build_preliminary_request(data, base_path):
    image_paths = [os.path.join(base_path, path) for path in data["image_path"]]
    input_str = "\n".join([f"Picture {i+1}: <img>{path}</img>" for i, path in enumerate(image_paths)])
    instruction = "You're a radiologist. Analyze medical images and generate preliminary diagnostic status."
    return InferRequest(messages=[{"role": "user", "content": json.dumps({"instruction": instruction, "input": input_str})}])

def build_detailed_request(data, base_path, preliminary_diagnosis):
    image_paths = [os.path.join(base_path, path) for path in data["image_path"]]
    input_str = "\n".join([f"Picture {i+1}: <img>{path}</img>" for i, path in enumerate(image_paths)])
    input_str += f"\ndiagnosis: {preliminary_diagnosis}"
    instruction = "You're a radiologist. Analyze medical images and generate detailed diagnostic reports with reference to the preliminary diagnosis."
    return InferRequest(messages=[{"role": "user", "content": json.dumps({"instruction": instruction, "input": input_str})}])

def main():
    # ===================== 你的路径已全部填入 =====================
    image_dir = "/mnt/iu_xray/images"
    annotation_path = "/mnt/iu_xray/annotation.json"
    lora_path = "/mnt/zzh/lora_models/qwen2-vl-instruct/v5-20250312-020819/checkpoint-3800/"
    output_jsonl = "output.jsonl"

    # 加载模型
    engine1 = load_model(lora_path)
    engine2 = load_model(lora_path)

    # 读取标注
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    # 清洗报告
    for part in annotation_data:
        for d in annotation_data[part]:
            d["report"] = clean_report(d["report"], "iu_xray")

    # 推理并保存
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for data in annotation_data["test"]:
            req1 = build_preliminary_request(data, image_dir)
            pre = infer(engine1, req1)
            req2 = build_detailed_request(data, image_dir, pre)
            detailed = infer(engine2, req2)

            f_out.write(json.dumps({
                "id": data["id"],
                "true_report": data["report"],
                "generated_report": detailed
            }, ensure_ascii=False) + "\n")

    print(f"第一步完成！生成报告已保存：{output_jsonl}")

if __name__ == "__main__":
    main()