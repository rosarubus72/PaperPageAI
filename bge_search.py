import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

# ==============================
# 🔹 1. 向量检索模块 (BGE)
# ==============================
class BGEInternalRetriever:
    def __init__(self, model_path):
        """初始化 BGE 模型和分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        self.sections = []  # 存储文档片段
        self.embeddings = None  # 存储嵌入向量

    def load_document(self, json_path):
        """加载论文 JSON 文件并提取章节"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for section in data.get('sections', []):
            title = section.get('title', '').strip()
            content = section.get('content', '').strip()
            if content and title not in ['References', 'Contents']:
                self.sections.append({
                    'title': title,
                    'content': content,
                    'text': f"{title}: {content}"
                })
        
        self._encode_sections()

    def _encode_sections(self):
        """批量生成嵌入"""
        texts = [sec['text'] for sec in self.sections]
        embeddings = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model(**inputs)
                batch_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_emb)
        self.embeddings = np.vstack(embeddings)

    def _encode_query(self, query):
        """编码查询语句"""
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model(**inputs)
            return output.last_hidden_state[:, 0, :].cpu().numpy()

    def retrieve(self, query, top_k=5):
        """检索最相关段落"""
        query_emb = self._encode_query(query)
        similarities = cosine_similarity(query_emb, self.embeddings).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        return [
            {
                'title': self.sections[i]['title'],
                'content': self.sections[i]['content'],
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]


# ==============================
# 🔹 2. LLM 模块 (Qwen2.5)
# ==============================
class QwenSummarizer:
    def __init__(self, model_path):
        """初始化 Qwen 模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype="auto"
        )
        self.model.eval()
        print("✅ Qwen model loaded successfully.")

    def generate_summary(self, module_name, query, retrieved_results, max_new_tokens=512):
        """根据检索结果生成模块文本"""
        context = ""
        for i, res in enumerate(retrieved_results, 1):
            context += f"【Section #{i}】\nTitle: {res['title']}\nContent: {res['content']}\n\n"

        prompt = f"""
You are an expert academic assistant helping to summarize a research paper for its online homepage.

Below are excerpts from the paper that may relate to the topic of **{module_name}**.

---
{context}
---

Now answer the following question concisely and academically:

{query}

Guidelines:
1. Base your answer only on the given excerpts.
2. Write in an academic yet accessible tone (for a project webpage).
3. Avoid unnecessary filler or repetition.
4. Keep length between 150–250 words.
5. Focus on clarity, coherence, and factual accuracy.
6. DO NOT generate Markdown headings (e.g., #, ##, ###) in your output.
7. You MAY use other Markdown formatting, such as bold (**text**) or italics (*text*).

Please generate the content for the "{module_name}" section directly.
"""

        messages = [
            {"role": "system", "content": "You are a scientific summarization assistant specialized in research paper interpretation."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        response = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        return response.strip()


# ==============================
# 🔹 3. 主逻辑：RAG + Agent
# ==============================
def main():
    # === 模型路径 ===
    BGE_MODEL_PATH = "/home/gaojuanru/.cache/huggingface/BAAI/bge-m3"
    QWEN_MODEL_PATH = "/home/gaojuanru/.cache/huggingface/Qwen/Qwen2.5-7B-Instruct"

    # === 论文 JSON 路径 ===
    JSON_PATH = "./jiexi/3D-MOOD_ Lifting 2D to 3D for Monocular Open-Set Object Detection/3D-MOOD_ Lifting 2D to 3D for Monocular Open-Set Object Detection_content.json"

    # === 初始化模型 ===
    retriever = BGEInternalRetriever(BGE_MODEL_PATH)
    retriever.load_document(JSON_PATH)

    summarizer = QwenSummarizer(QWEN_MODEL_PATH)

    # === 四个模块问题 ===
    queries = {
        "Motivation": (
            "What problem or limitation in existing methods motivates this research? "
            "Why is addressing this problem important or challenging?"
        ),
        "Innovation": (
            "What are the main innovations and key contributions of this paper? "
            "What makes the proposed approach different from previous works?"
        ),
        "Methodology": (
            "How does the proposed method work? "
            "Describe the core architecture, main modules, and the process by which it solves the problem."
        ),
        "Experiments": (
            "What experiments were conducted and what are the key findings? "
            "Summarize how the results demonstrate the effectiveness or advantages of the method."
        )
    }

    summaries = {}

    for module_name, query in queries.items():
        print(f"\n===== 🧩 Generating section: {module_name} =====")
        results = retriever.retrieve(query, top_k=5)

        print(f"🔍 Retrieved {len(results)} relevant sections for {module_name}.")
        summary = summarizer.generate_summary(module_name, query, results)

        summaries[module_name] = {
            "query": query,
            "summary": summary,
            "retrieved_sections": results
        }

        print(f"✅ {module_name} summary generated.\n")

    # === 保存结果 ===
    output_path = JSON_PATH.replace(".json", "_modules.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 All module summaries saved to: {output_path}")
    print("\n===== 📘 Summary Overview =====")
    for name, data in summaries.items():
        print(f"\n### {name}\n{data['summary']}\n")


# if __name__ == "__main__":
#     main()
# ==============================
# 🔹 批量处理所有论文文件夹
# ==============================
def batch_generate_all():
    base_dir = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/jiexi"
    BGE_MODEL_PATH = "/home/gaojuanru/.cache/huggingface/BAAI/bge-m3"
    QWEN_MODEL_PATH = "/home/gaojuanru/.cache/huggingface/Qwen/Qwen2.5-7B-Instruct"

    # 初始化模型（加载一次节省时间）
    retriever = BGEInternalRetriever(BGE_MODEL_PATH)
    summarizer = QwenSummarizer(QWEN_MODEL_PATH)

    # 遍历每个子论文文件夹
    for paper_dir in sorted(os.listdir(base_dir)):
        paper_path = os.path.join(base_dir, paper_dir)
        if not os.path.isdir(paper_path):
            continue

        # 查找 *_content.json 文件
        content_files = [f for f in os.listdir(paper_path) if f.endswith("_content.json")]
        if not content_files:
            print(f"⚠️ No content.json found in {paper_dir}, skipped.")
            continue

        json_path = os.path.join(paper_path, content_files[0])
        output_path = json_path.replace("_content.json", "_content_modules.json")

        print(f"\n🚀 Processing paper: {paper_dir}")
        print(f"📄 Input: {json_path}")

        try:
            # 加载论文
            retriever.sections = []
            retriever.embeddings = None
            retriever.load_document(json_path)

            # 四个模块问题
            queries = {
                "Motivation": (
                    "What problem or limitation in existing methods motivates this research? "
                    "Why is addressing this problem important or challenging?"
                ),
                "Innovation": (
                    "What are the main innovations and key contributions of this paper? "
                    "What makes the proposed approach different from previous works?"
                ),
                "Methodology": (
                    "How does the proposed method work? "
                    "Describe the core architecture, main modules, and process by which it solves the problem."
                ),
                "Experiments": (
                    "What experiments were conducted and what are the key findings? "
                    "Summarize how the results demonstrate the effectiveness or advantages of the method."
                )
            }

            summaries = {}

            for module_name, query in queries.items():
                print(f"\n===== 🧩 Generating section: {module_name} =====")
                results = retriever.retrieve(query, top_k=5)
                summary = summarizer.generate_summary(module_name, query, results)
                summaries[module_name] = {
                    "query": query,
                    "summary": summary,
                    "retrieved_sections": results
                }
                print(f"✅ {module_name} done.")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, ensure_ascii=False, indent=2)

            print(f"🎯 Saved to {output_path}\n")

        except Exception as e:
            print(f"❌ Failed on {paper_dir}: {e}")

    print("\n✅ All papers processed successfully.")


if __name__ == "__main__":
    batch_generate_all()

