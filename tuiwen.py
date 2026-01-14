import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import shutil
import pandas as pd
from llms.llm2 import LLM2

# ==========================================================
# 🔹 论文讲解内容生成器（替代原TweetContentAgent）
# ==========================================================
class PaperExplanationAgent:
    """生成结构化、专业的论文讲解内容（Markdown格式）"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
    
    def generate_paper_title(self, paper_title, abstract):
        """生成吸引人的论文讲解标题（简洁、有亮点）"""
        prompt = f"""Generate an attractive title for a research paper explanation (Markdown format). Requirements:
        1. MUST include 1-2 relevant emojis at the beginning
        2. Keep it concise (10-20 words)
        3. Highlight core contribution/innovation of the paper
        4. Use professional but engaging tone
        5. Add a wave symbol ~ at the end
        
        Paper Title: {paper_title}
        Abstract: {abstract[:500]}
        
        Generate ONLY the title (no explanations, no quotes):
        """
        
        response = self.llm_model.generate(query=prompt)
        title = response.strip().strip('"').strip("'")
        
        # 确保有emoji，如果没有则添加
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        if not emoji_pattern.search(title):
            # 根据论文主题添加合适的emoji
            if any(word in paper_title.lower() for word in ['ai', 'machine', 'deep', 'neural']):
                title = f"🤖 {title}～"
            elif any(word in paper_title.lower() for word in ['vision', 'image', 'video', '3d']):
                title = f"👁️ {title}～"
            elif any(word in paper_title.lower() for word in ['language', 'text', 'nlp']):
                title = f"🗣️ {title}～"
            elif any(word in paper_title.lower() for word in ['learning', 'model', 'algorithm']):
                title = f"🧠 {title}～"
            else:
                title = f"🔬 {title}～"
        
        # 确保结尾有～
        if not title.endswith('～'):
            title = f"{title}～"
            
        return title
    
    def generate_paper_section(self, section_name, content, previously_generated=None):
        """生成结构化的论文讲解内容（专业、简洁）"""
        
        # 章节emoji映射（简洁版）
        section_emojis = {
            "abstract": "🔍 Abstract",
            "motivation": "🚀 Motivation",
            "innovation": "💡 Innovation",
            "methodology": "🛠️ Methodology",
            "experiments": "📊 Experiments"
        }
        
        section_header = section_emojis.get(section_name.lower(), f"📝 {section_name.capitalize()}")
        
        # 构建避免重复的上下文
        context_info = ""
        if previously_generated:
            context_info = f"""
**ALREADY COVERED (DO NOT REPEAT):**
{self._summarize_previous_content(previously_generated)}

**CRITICAL: Introduce NEW information not covered above.**
"""
        
        prompt = f"""Generate professional, structured explanation for the "{section_name}" section of a research paper (Markdown format). Requirements:

**STYLE GUIDELINES (MUST FOLLOW):**
1. Professional and academic tone, no casual/conversational language
2. Use clear, concise paragraphs and bullet points (-/1.) for key points
3. Avoid exclamation marks, rhetorical questions, and overly emotional language
4. Focus on factual, objective explanation of the paper's content
5. Use proper terminology, highlight key terms with **bold**
6. Length: 200-300 words, well-organized
7. Do NOT add hashtags, emojis (except in section header), or tweet-style language

{context_info}

**Source Content:**
{content[:1000]}

Generate ONLY the section content (no section title, no markdown headers, just paragraphs/bullet points):
"""
        
        try:
            response = self.llm_model.generate(query=prompt)
            cleaned = self._clean_section_content(response)
            return cleaned
        except Exception as e:
            print(f"❌ 生成章节内容失败: {e}")
            return self._fallback_section_content(content, section_name)
    
    def _clean_section_content(self, content):
        """清理生成的章节内容"""
        # 移除可能的提示词残留
        content = re.sub(r'^(请生成|生成内容|内容:|#+)\s*', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        # 移除多余的空行，保留合理的分段
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        # 规范化列表符号
        content = re.sub(r'^(\d+)\. ', r'1. ', content, flags=re.MULTILINE)
        content = re.sub(r'^- ', r'- ', content, flags=re.MULTILINE)
        
        return content
    
    def _summarize_previous_content(self, previous_content):
        """总结已生成的内容"""
        summary = []
        for section_name, content in previous_content.items():
            # 提取核心信息
            sentences = re.split(r'[.!?]', content)
            first_sentence = sentences[0].strip() if sentences else ""
            if first_sentence:
                summary.append(f"- {section_name}: {first_sentence}")
        
        return "\n".join(summary) if summary else "No previous content."
    
    def _fallback_section_content(self, content, section_name):
        """备用方法生成章节内容"""
        # 简化内容，结构化展示
        sentences = re.split(r'[.!?]', content)
        key_points = [s.strip() for s in sentences if s.strip()][:4]
        
        # 构建结构化内容
        section_text = ""
        if section_name in ["motivation", "innovation", "methodology", "experiments"]:
            # 使用列表形式
            section_text = "\n".join([f"- {point}." for point in key_points])
        else:
            # 段落形式
            section_text = " ".join(key_points) + "."
        
        return section_text

# ==========================================================
# 🔹 视觉内容选择器（简化版）
# ==========================================================
class VisualSelector:
    """为论文讲解选择合适的视觉内容"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
    
    def select_visuals(self, section_name, section_content, candidates, max_per_section=1):
        """为章节选择最合适的视觉内容"""
        if not candidates:
            return []
        
        candidate_text = ""
        for i, (v, score) in enumerate(candidates, 1):
            vtype = "Table" if "table_path" in v else "Figure"
            cap = v.get("caption", "")
            candidate_text += f"[{i}] ({vtype}, relevance={score:.3f}) {cap[:100]}...\n"
        
        prompt = f"""Select the most relevant visual for a research paper explanation. Requirements:

**SECTION:** {section_name}
**SECTION CONTENT:** {section_content[:300]}...

**SELECTION CRITERIA:**
1. RELEVANCE: Choose visuals that directly relate to the section content
2. CLARITY: Prefer clear, easy-to-understand visuals
3. INFORMATION VALUE: Select visuals that enhance understanding of key concepts/results

**VISUAL CANDIDATES:**
{candidate_text}

**SECTION GUIDELINES:**
- For Abstract/Motivation: Choose conceptual diagrams, problem illustrations
- For Innovation: Choose novel framework diagrams, comparison visuals
- For Methodology: Choose clean architecture diagrams, process flows
- For Experiments: Choose key results tables, performance charts

Return ONLY a JSON list of indices, e.g., [1] or []
Maximum {max_per_section} visual per section.
"""
        
        try:
            resp = self.llm_model.generate(query=prompt)
            matched = json.loads(re.search(r"\[.*?\]", resp, re.S).group())
            return [candidates[i-1][0] for i in matched if 1 <= i <= len(candidates)]
        except:
            # 默认选择相关性最高的1个
            return [candidates[0][0]] if candidates else []

# ==========================================================
# 🔹 BGE 模型（保持不变）
# ==========================================================
class BGEEmbedder:
    def __init__(self, model_path):
        device = torch.device("cuda:2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path, dtype=torch.float16, device_map={"": device}
        )
        self.model.eval()

    def encode(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=512
            ).to(self.model.device)
            with torch.no_grad():
                output = self.model(**inputs)
                batch_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_emb)
        return np.vstack(embeddings)

    def similarity(self, query, candidates, top_k=5):
        q_emb = self.encode([query])
        c_emb = self.encode([c["text"] for c in candidates])
        sims = cosine_similarity(q_emb, c_emb).flatten()
        top_indices = sims.argsort()[::-1][:top_k]
        return [(candidates[i], float(sims[i])) for i in top_indices]

# ==========================================================
# 🔹 论文讲解生成器主类
# ==========================================================
class PaperExplanationGenerator:
    def __init__(self, content_json_path, modules_json_path, 
                 images_json_path=None, tables_json_path=None, csv_path=None):
        self.content_json_path = content_json_path
        self.modules_json_path = modules_json_path
        self.images_json_path = images_json_path
        self.tables_json_path = tables_json_path
        self.csv_path = csv_path

        # 加载数据
        self.paper_content = self._load_json(content_json_path)
        self.modules_content = self._load_json(modules_json_path)
        self.images_data = self._load_json(images_json_path) or {}
        self.tables_data = self._load_json(tables_json_path) or {}
        
        # 初始化LLM
        self.llm_model = LLM2('Qwen2.5-7B-Instruct')
        
        # 初始化代理（替换为新的生成器）
        self.paper_agent = PaperExplanationAgent(self.llm_model)
        self.visual_selector = VisualSelector(self.llm_model)
        self.bge = BGEEmbedder("/mnt/gaojuanru/twittergenerate/cache/huggingface/BAAI/bge-m3")
        
        self.used_visuals = set()
        self.output_assets_dir = None
        self.assets_mapping = {}
        
        # 规划的章节（固定）
        self.planned_sections = ["abstract", "motivation", "innovation", "methodology", "experiments"]
    
    def _load_json(self, path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _extract_paper_info(self):
        """提取论文基本信息"""
        sections = self.paper_content.get("sections", [])
        title = sections[0].get("title", "Untitled Paper") if sections else "Untitled Paper"
        
        # 查找摘要
        abstract = ""
        for section in sections:
            if "abstract" in section.get("title", "").lower():
                abstract = section.get("content", "")
                break
            elif section.get("content", "").strip():
                abstract = section.get("content", "")[:500]
                break
        
        # 提取作者信息
        authors = []
        if sections:
            first_content = sections[0].get("content", "")
            # 简单的作者提取
            author_patterns = [
                r'([A-Z][a-zA-Z\-\.]+\s+[A-Z][a-zA-Z\-\.]+)',
                r'([A-Z]\.\s*[A-Z][a-zA-Z\-]+)'
            ]
            for pattern in author_patterns:
                matches = re.findall(pattern, first_content)
                if matches:
                    authors.extend(matches)
                    if len(authors) >= 3:  # 最多取3个作者
                        break
        
        return {
            "title": title,
            "abstract": abstract[:1000] if abstract else "No abstract available.",
            "authors": authors[:3] if authors else ["Anonymous"],
            "sections": sections
        }
    
    def _get_section_content(self, section_name):
        """获取章节内容用于生成"""
        section_name_lower = section_name.lower()
        
        # 1. 从modules_content查找
        for module_name, module_data in self.modules_content.items():
            if section_name_lower in module_name.lower():
                return module_data.get("summary", "")
        
        # 2. 从paper_content查找
        for section in self.paper_content.get("sections", []):
            if section_name_lower in section.get("title", "").lower():
                return section.get("content", "")
        
        # 3. 使用摘要作为后备
        paper_info = self._extract_paper_info()
        return paper_info["abstract"]
    
    def _select_visuals_for_section(self, section_name, section_content):
        """为章节选择视觉内容"""
        candidates = []
        
        # 从图片数据中提取候选
        for fig in self.images_data.values():
            if fig.get("image_path") and fig.get("image_path") not in self.used_visuals:
                candidates.append({
                    "text": fig.get("caption", ""),
                    "image_path": fig.get("image_path"),
                    "caption": fig.get("caption", ""),
                    "type": "image"
                })
        
        # 从表格数据中提取候选（作为图片处理）
        for tb in self.tables_data.values():
            if tb.get("table_path") and tb.get("table_path") not in self.used_visuals:
                candidates.append({
                    "text": tb.get("caption", "") + "\n" + tb.get("table_text", ""),
                    "image_path": tb.get("table_path"),
                    "caption": tb.get("caption", ""),
                    "type": "table"
                })
        
        if not candidates:
            return []
        
        # 使用BGE计算相关性
        top_candidates = self.bge.similarity(section_content, candidates, top_k=3)
        
        # 选择视觉内容
        selected = self.visual_selector.select_visuals(
            section_name, section_content, top_candidates, max_per_section=1
        )
        
        # 标记已使用
        for v in selected:
            self.used_visuals.add(v.get("image_path"))
        
        return selected[:1]  # 最多1个视觉内容
    
    def _copy_asset(self, original_path):
        """复制资源文件到输出目录"""
        if not original_path or not os.path.exists(original_path):
            return None
            
        if original_path in self.assets_mapping:
            return self.assets_mapping[original_path]
        
        original_file = Path(original_path)
        new_filename = f"section_{len(self.assets_mapping)}_{original_file.name}"
        relative_path = f"assets/{new_filename}"
        
        destination = self.output_assets_dir / new_filename
        try:
            shutil.copy2(original_path, destination)
            self.assets_mapping[original_path] = relative_path
            print(f"✅ 复制资源: {original_path} -> {relative_path}")
            return relative_path
        except Exception as e:
            print(f"❌ 复制资源失败: {original_path}, 错误: {e}")
            return None
    
    def _generate_markdown(self, paper_data):
        """生成结构化的Markdown格式论文讲解（核心修改）"""
        # 主标题
        md_content = f"# {paper_data['title']}\n\n"
        
        # 生成各个章节内容（结构化）
        section_emojis = {
            "abstract": "🔍 Abstract",
            "motivation": "🚀 Motivation",
            "innovation": "💡 Innovation",
            "methodology": "🛠️ Methodology",
            "experiments": "📊 Experiments"
        }
        
        for section in self.planned_sections:
            if section in paper_data['sections']:
                section_data = paper_data['sections'][section]
                section_header = section_emojis.get(section.lower(), section.capitalize())
                
                # 添加章节标题
                md_content += f"### {section_header}\n"
                # 添加章节内容
                md_content += f"{section_data['content']}\n\n"
                
                # 添加视觉内容（简化版）
                for visual in section_data.get('visuals', []):
                    if visual.get('relative_path'):
                        caption = visual.get('caption', '').strip()
                        # 清理标题
                        caption = re.sub(r'^(Figure|Fig\.|Table|Tab\.)\s*\d+[\.:]\s*', '', caption, flags=re.IGNORECASE)
                        md_content += f"![{caption[:100]}]({visual['relative_path']})\n"
                        if caption:
                            md_content += f"> {caption}\n\n"
        
        return md_content
    
    def generate_explanation(self, output_path):
        """生成论文讲解Markdown"""
        print(f"🚀 开始生成论文讲解...")
        
        # 设置输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建assets目录
        self.output_assets_dir = output_dir / "assets"
        self.output_assets_dir.mkdir(exist_ok=True)
        self.assets_mapping = {}
        
        # 提取论文信息
        print("📋 提取论文信息...")
        paper_info = self._extract_paper_info()
        
        # 生成标题
        print("🎯 生成讲解标题...")
        paper_title = self.paper_agent.generate_paper_title(
            paper_info["title"], 
            paper_info["abstract"]
        )
        
        # 生成各个部分的内容
        print("📝 生成讲解内容...")
        paper_sections = {}
        previously_generated = {}
        
        for section in self.planned_sections:
            print(f"  - 处理: {section}")
            
            # 获取基础内容
            base_content = self._get_section_content(section)
            
            # 生成结构化内容
            section_content = self.paper_agent.generate_paper_section(
                section, 
                base_content,
                previously_generated
            )
            
            # 选择视觉内容
            visuals = self._select_visuals_for_section(section, section_content)
            
            # 复制视觉资源
            visual_data = []
            for visual in visuals:
                relative_path = self._copy_asset(visual.get("image_path"))
                if relative_path:
                    visual_data.append({
                        "relative_path": relative_path,
                        "caption": visual.get("caption", "")
                    })
            
            # 存储该部分数据
            paper_sections[section] = {
                "content": section_content,
                "visuals": visual_data
            }
            
            # 记录已生成内容
            previously_generated[section] = section_content
        
        # 构建论文数据
        paper_data = {
            "title": paper_title,
            "authors": paper_info["authors"],
            "sections": paper_sections
        }
        
        # 生成Markdown
        print("🔄 生成Markdown文件...")
        markdown_content = self._generate_markdown(paper_data)
        
        # 写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"✅ 论文讲解生成完成: {output_path}")
        print(f"📁 资源文件保存在: {self.output_assets_dir}")
        print(f"📊 共复制了 {len(self.assets_mapping)} 个资源文件")
        
        return output_path

# ==========================================================
# 🔹 批量生成模式
# ==========================================================
if __name__ == "__main__":
    # 设置基础路径
    base_dir = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/jiexi"
    output_base_dir = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/paper_explanation_output"
    
    # 遍历所有论文子文件夹
    for subdir in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        
        # 自动匹配JSON文件
        content_json = os.path.join(sub_path, f"{subdir}_content.json")
        modules_json = os.path.join(sub_path, f"{subdir}_content_modules.json")
        images_json = os.path.join(sub_path, f"{subdir}_images.json")
        tables_json = os.path.join(sub_path, f"{subdir}_tables.json")
        
        # 确保至少有content和modules文件
        if not (os.path.exists(content_json) and os.path.exists(modules_json)):
            print(f"⚠️ 缺少主要JSON文件: {subdir}, 跳过。")
            continue
        
        # 创建输出目录
        output_dir = os.path.join(output_base_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 输出文件路径
        output_md = os.path.join(output_dir, f"paper_explanation_{subdir}.md")
        
        print(f"\n🚀 正在为 {subdir} 生成论文讲解...")
        
        try:
            # 初始化生成器（替换为新的类）
            generator = PaperExplanationGenerator(
                content_json_path=content_json,
                modules_json_path=modules_json,
                images_json_path=images_json if os.path.exists(images_json) else None,
                tables_json_path=tables_json if os.path.exists(tables_json) else None
            )
            
            # 生成讲解
            generator.generate_explanation(output_md)
            print(f"✅ {subdir} 论文讲解生成成功")
            
        except Exception as e:
            print(f"❌ {subdir} 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 所有论文讲解生成完成!")