import os
import dashscope
import json
from datetime import datetime

def load_tags_data(tags_file_path):
    id_to_tags = {}
    try:
        if not os.path.exists(tags_file_path):
            print(f"错误：Tags文件不存在！路径：{tags_file_path}")
            return id_to_tags

        with open(tags_file_path, 'r', encoding='utf-8-sig') as f:
            first_char = f.read(1).strip()
            f.seek(0)
            raw_content = f.read().strip()

        if first_char != '[':
            print("错误：文件最外层不是列表！")
            return id_to_tags

        try:
            tags_data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            print(f"JSON格式错误：{e.msg}")
            return id_to_tags

        print(f"成功读取Tags文件，共{len(tags_data)}条数据")
        for idx, item in enumerate(tags_data, 1):
            if not isinstance(item, dict):
                continue
            report_id = item.get("id")
            tags = item.get("Tags", [])
            if not report_id or not isinstance(tags, list):
                continue
            clean_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
            id_to_tags[report_id] = clean_tags

        print(f"Tags加载完成！共{len(id_to_tags)}条有效记录")
        return id_to_tags

    except Exception as e:
        print(f"加载Tags错误：{str(e)}")
        return id_to_tags


def optimize_report(query_info, similar_reports, id_to_tags, api_key, model="qwen-plus-latest"):
    original_report = query_info["original_generated_report"]
    query_id = query_info["query_id"]

    print("\n--- Tags匹配检查 ---")
    if query_id in id_to_tags:
        tags = id_to_tags[query_id]
        print(f"成功匹配到Tags！ID：{query_id}，标签：{tags}")
        tags_str = ", ".join(tags)
    else:
        sample_ids = list(id_to_tags.keys())[:3]
        print(f"未匹配到Tags！query_id：{query_id}")
        print(f"Tags文件中前3个id示例：{sample_ids}")
        tags_str = "No specific tags (ID not matched)"

    similar_reports_str = "No similar report"
    if similar_reports:
        s = similar_reports[0]
        similar_reports_str = s['cleaned_train_report']

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
Key MeSH terms (core disease labels to emphasize): {tags_str}

Optimize the intermediate report by following the above rules. Output only the optimized report.
            """
        }
    ]

    try:
        response = dashscope.Generation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            result_format='message',
            temperature=0.3
        )
        if response and response.output and response.output.choices:
            return response.output.choices[0].message.content.strip() or original_report
        else:
            print(f"API返回异常，query_id: {query_id}")
            return original_report
    except Exception as e:
        print(f"API调用错误 for query_id {query_id}: {e}")
        return original_report


def main():
    # ===================== 你的 MIMIC 路径已全部配置 =====================
    json_file_path = "retrieval_results_mimic.json"          # 第二步输出
    tags_file_path = "/mnt/wqw/Data/mimic_all_MeSH_Tags_test_partition.json"  # MIMIC Tags
    api_key = ""           # 你的API
    final_save = "final_optimized_report_mimic.json"         # 最终输出

    os.makedirs("data", exist_ok=True)

    # 加载Tags
    id_to_tags = load_tags_data(tags_file_path)

    # 读取检索结果
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"\n开始优化 MIMIC-CXR 报告，总计：{total} 条\n")

    for i, item in enumerate(data, 1):
        query_info = item["query_info"]
        similar = item["similar_train_reports"]

        opt = optimize_report(query_info, similar, id_to_tags, api_key)
        item["optimized_report"] = opt

        if i % 10 == 0:
            print(f"已处理：{i}/{total}")

    # 保存最终报告
    with open(final_save, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n MIMIC 全部优化完成！最终文件：{final_save}")


if __name__ == "__main__":
    main()