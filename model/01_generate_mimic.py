import os
import json
import re
import torch
from swift.llm import (
    PtEngine, RequestConfig, BaseArguments, InferRequest, safe_snapshot_download
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def clean_report(report, dataset="mimic_cxr"):
    if dataset == "mimic_cxr":
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('  ', ' ') \
            .replace('..', '.').replace('1. ', '').replace('. 2. ', '. ').replace(':',' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
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
    return PtEngine(args.model, adapters=[adapter_path], lazy_tokenize=True)

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
    image_dir = "/mnt/mimic_cxr/images/"
    annotation_path = "/mnt/mimic_cxr/test.json"
    lora_path = "/mnt/zzh/lora_models/qwen2-vl-instruct/mimic/v1-20250319-124949/checkpoint-400/"
    output_jsonl = "output_mimic.jsonl"

    engine = load_model(lora_path)

    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)

    for part in annotation_data:
        for d in annotation_data[part]:
            d["report"] = clean_report(d["report"], "mimic_cxr")

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for data in annotation_data["test"]:
            req1 = build_preliminary_request(data, image_dir)
            pre = infer(engine, req1)
            req2 = build_detailed_request(data, image_dir, pre)
            detailed = infer(engine, req2)

            f_out.write(json.dumps({
                "id": data["id"],
                "true_report": data["report"],
                "generated_report": detailed
            }, ensure_ascii=False) + "\n")

    print(f"MIMIC 生成完成：{output_jsonl}")

if __name__ == "__main__":
    main()