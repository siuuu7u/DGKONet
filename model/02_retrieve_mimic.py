import json
import re
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def clean_report(report, dataset="mimic_cxr"):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('  ', ' ') \
        .replace('..', '.').replace('1. ', '').replace('. 2. ', '. ').replace(':',' :') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
    return ' . '.join(tokens) + ' .'

class ReportRetriever:
    def __init__(self, embedding_model, dataset_type="mimic_cxr"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.train_reports = []
        self.cleaned_train_reports = []
        self.dataset_type = dataset_type

    def load_train_data(self, train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
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
    bge_path = "/mnt/zzh/models/bge-base-en-v1.5"
    train_json = "/mnt/mimic_cxr/annotation.json"
    input_jsonl = "output_mimic.jsonl"
    output_result = "retrieval_results_mimic.json"

    retriever = ReportRetriever(bge_path)
    retriever.load_train_data(train_json)
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

    print(f"MIMIC 检索完成：{output_result}")