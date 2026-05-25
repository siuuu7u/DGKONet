import os
import dashscope
import json
from datetime import datetime

def load_mesh(mesh_path):
    try:
        with open(mesh_path, encoding='utf-8') as f:
            data = json.load(f)
        mapping = {}
        for part in data:
            for item in data[part]:
                mapping[item["id"]] = item["MeSH"]
        return mapping
    except:
        return {}

def optimize_report(query_info, similar_reports, id2mesh, api_key):
    original = query_info["original_generated_report"]
    sim = similar_reports[0]
    train_id = sim["train_id"]
    mesh_str = ", ".join(id2mesh.get(train_id, []))

    messages = [
        {
            "role": "system",
            "content": """
You are a professional radiology report optimization expert. Optimize the intermediate report based on the intermediate report, similar reference reports, and key MeSH terms (core disease labels). Follow these strict rules:

1. Preserve all core diagnostic information (normal/abnormal conclusions, key findings) without altering medical judgment.
2. Use the MeSH term as a guideline for emphasizing core disease labels to ensure the accuracy of its expression and consistency with radiological practices.
3. Learn professional expression style from similar reports: terminology accuracy, sentence structure, logical coherence.
4. Correct redundant/non-standard expressions, but keep the report concise and avoid adding new information.
5. Output only the optimized report text (no explanations).
            """
        },
        {
            "role": "user",
            "content": f"""
Intermediate report: {original_report}
Similar reference reports: {similar_reports_str}
Key MeSH terms (core disease labels to emphasize): {mesh_str}

Optimize the intermediate report by following the above rules. Output only the optimized report.
            """
        }
    ]

    try:
        rsp = dashscope.Generation.call(
            api_key=api_key, model="qwen-plus-latest", messages=messages, temperature=0.3
        )
        return rsp.output.choices[0].message.content.strip()
    except:
        return original

if __name__ == "__main__":
    # ===================== 你的路径已全部填入 =====================
    retrieval_file = "retrieval_results.json"
    mesh_file = "/mnt/wqw/Data/iu_xray/iu_all_MeSH.json"
    api_key = ""
    final_output = "final_optimized_report.json"

    id2mesh = load_mesh(mesh_file)
    with open(retrieval_file, encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        opt = optimize_report(item["query_info"], item["similar_train_reports"], id2mesh, api_key)
        item["optimized_report"] = opt

    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"全部完成！最终报告：{final_output}")