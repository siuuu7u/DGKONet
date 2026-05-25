import json
import re
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def clean_report(report, dataset="iu_xray"):
    if dataset == "iu_xray":
        report_cleaner = lambda t: t.replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
        return ' . '.join(tokens) + ' .'
    return report

class ReportRetriever:
    def __init__(self, embedding_model, dataset_type="iu_xray"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.train_reports = []
        self.cleaned_train_reports = []
        self.dataset_type = dataset_type

    def load_train_data(self, annotation_json):
        with open(annotation_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.train_reports = data["train"]
        self.cleaned_train_reports = [clean_report(it["report"]) for it in self.train_reports]

    def build_index(self):
        embeddings = self.model.encode(self.cleaned_train_reports, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_top1(self, query_text):
        cleaned = clean_report(query_text)
        emb = self.model.encode(cleaned, convert_to_numpy=True).reshape(1, -1)
        scores, idx = self.index.search(emb, 1)
        return self.train_reports[idx[0][0]], float(scores[0][0]), self.cleaned_train_reports[idx[0][0]]

if __name__ == "__main__":
    # ===================== 你的路径已全部填入 =====================
    annotation_json = "/mnt/iu_xray/annotation.json"
    bge_model_path = "/mnt/zzh/models/bge-base-en-v1.5"
    input_jsonl = "output.jsonl"
    output_result = "retrieval_results.json"

    retriever = ReportRetriever(bge_model_path)
    retriever.load_train_data(annotation_json)
    retriever.build_index()

    results = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            report, score, cleaned = retriever.retrieve_top1(item["generated_report"])
            results.append({
                "query_info": {
                    "query_id": item["id"],
                    "original_generated_report": item["generated_report"],
                    "true_report": item["true_report"]
                },
                "similar_train_reports": [{
                    "train_id": report["id"],
                    "cleaned_train_report": cleaned,
                    "similarity_score": score
                }]
            })

    with open(output_result, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"第二步完成！检索结果已保存：{output_result}")