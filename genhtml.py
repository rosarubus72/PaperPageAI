import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
import shutil
import pandas as pd
from llms.llm2 import LLM2
from bs4 import BeautifulSoup
from bs4 import Comment
# ==========================================================
# 🔹 简化的表格解析器 - 专注于VLM生成
# ==========================================================
import html
import re

class PlannerAgent:
    def __init__(self, llm_model):
        self.llm_model = llm_model
    
    def plan_homepage_sections(self, paper_content, modules_content):
        """根据论文内容规划主页应该展示哪些部分，确保连贯性"""
        # 直接返回模板中固定的部分
        fixed_sections = ["abstract", "motivation", "innovation", "methodology", "experiments"]
        print(f"✅ PlannerAgent返回固定部分: {fixed_sections}")
        return fixed_sections
    
    def _parse_plan_response(self, resp):
        """解析规划响应 - 备用方法"""
        # 返回固定部分
        return ["abstract", "motivation", "innovation", "methodology", "experiments"]

# ==========================================================
# 🔹 文本格式化工具
# ==========================================================
class TextFormatter:
    @staticmethod
    def clean_caption(caption):
        """清理图表标题，移除Figure X、Table X等前缀"""
        if not caption:
            return ""
        
        # 常见的图表前缀模式
        patterns = [
            r'^Figure\s*\d+[\.:]\s*',      # Figure 1: 或 Figure 1.
            r'^Fig\.\s*\d+[\.:]\s*',       # Fig. 1: 或 Fig. 1.
            r'^Table\s*\d+[\.:]\s*',       # Table 1: 或 Table 1.
            r'^Tab\.\s*\d+[\.:]\s*',       # Tab. 1: 或 Tab. 1.
            r'^FIG\.\s*\d+[\.:]\s*',       # FIG. 1: 或 FIG. 1.
            r'^TABLE\s*\d+[\.:]\s*',       # TABLE 1: 或 TABLE 1.
            r'^Fig\s*\d+[\.:]\s*',         # Fig 1: 或 Fig 1.
            r'^Tab\s*\d+[\.:]\s*',         # Tab 1: 或 Tab 1.
        ]
        
        cleaned_caption = caption.strip()
        for pattern in patterns:
            # 尝试匹配并移除前缀
            cleaned_caption = re.sub(pattern, '', cleaned_caption, flags=re.IGNORECASE)
        
        # 如果清理后为空，返回原始标题
        if not cleaned_caption.strip():
            return caption.strip()
        
        # 确保首字母大写
        cleaned_caption = cleaned_caption.strip()
        if cleaned_caption and cleaned_caption[0].islower():
            cleaned_caption = cleaned_caption[0].upper() + cleaned_caption[1:]
        
        return cleaned_caption
    
    @staticmethod
    def format_text(text):
        """处理加粗、列表、重点颜色等格式转换"""
        if not text:
            return ""
            
        # 1. 转换加粗为<strong>标签
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # 2. 转换数字列表（1. 2. 3.）为有序列表
        lines = text.split('\n')
        in_list = False
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                if not in_list:
                    formatted_lines.append('<ol class="list-decimal pl-6 space-y-2">')
                    in_list = True
                # 提取列表内容并保留格式
                content = re.sub(r'^\d+\.\s*', '', line)
                formatted_lines.append(f'  <li>{content}</li>')
            else:
                if in_list:
                    formatted_lines.append('</ol>')
                    in_list = False
                formatted_lines.append(line)
        if in_list:
            formatted_lines.append('</ol>')
        text = '\n'.join(formatted_lines)
        
        # 3. 为特定关键词添加颜色
        keywords = [
            r'PosterAgent', r'Qwen', r'GPT-4o',  # 模型名
            r'Visual Quality', r'Textual Coherence', r'PaperQuiz'  # 指标名
        ]
        for kw in keywords:
            text = re.sub(
                fr'({kw})',
                r'<span class="text-primary font-semibold">\1</span>',
                text,
                flags=re.IGNORECASE
            )
        
        return text

# ==========================================================
# 🔹 BGE 模型
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
# 🔹 Qwen 决策器
# ==========================================================
class QwenDecider:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        print("✅ Qwen2.5-7B-Instruct loaded via LLM2.")
    
    def _get_module_specific_guidance(self, module_name):
        """为特定模块提供视觉选择指导"""
        guidance_map = {
            "introduction": """
        - PRIORITY: Conceptual diagrams, problem illustrations, motivation figures
        - ACCEPTABLE: High-level architecture overviews (if essential for context)
        - AVOID: Detailed experimental results, technical parameter tables
        - RESERVE: All quantitative results for experiments sections""",

            "background": """
        - PRIORITY: Comparative analysis with prior work, domain context figures
        - ACCEPTABLE: Foundational concept illustrations
        - AVOID: Novel methodology diagrams, experimental results
        - RESERVE: Your technical innovations for methodology section""",

            "methodology": """
        - PRIORITY: Architecture diagrams, process flows, algorithm visualizations
        - ACCEPTABLE: Parameter tables, component specifications
        - AVOID: Experimental results, performance comparisons
        - RESERVE: All results for dedicated experiments sections""",

            "method": """
        - PRIORITY: Architecture diagrams, process flows, algorithm visualizations
        - ACCEPTABLE: Parameter tables, component specifications
        - AVOID: Experimental results, performance comparisons
        - RESERVE: All results for dedicated experiments sections""",

            "experiments": """
        - PRIORITY: Results tables, performance charts, ablation studies
        - ACCEPTABLE: Comparison figures, statistical analyses
        - AVOID: Conceptual diagrams, architecture overviews
        - USE NOW: This is the primary section for experimental visuals""",

            "results": """
        - PRIORITY: Quantitative results, evaluation metrics, benchmark comparisons
        - ACCEPTABLE: Visualization of findings, statistical significance
        - AVOID: Methodological diagrams, conceptual illustrations
        - USE NOW: Critical results should be displayed here""",

            "innovation": """
        - PRIORITY: Novel framework diagrams, comparison with existing methods
        - ACCEPTABLE: Technical novelty illustrations
        - AVOID: Detailed experimental results
        - RESERVE: Results for experiments sections""",

            "motivation": """
        - PRIORITY: Problem illustrations, motivation diagrams
        - ACCEPTABLE: High-level overviews
        - AVOID: Technical details, experimental results
        - RESERVE: Technical diagrams for methodology section"""
        }
        
        # 查找匹配的指导
        module_lower = module_name.lower()
        for key, guidance in guidance_map.items():
            if key in module_lower:
                return guidance
        
        # 默认指导
        return """
        - Assess if this module is primarily: conceptual, methodological, experimental, or summary-oriented
        - Reserve experimental results for experiments/results sections
        - Save technical diagrams for methodology/approach sections  
        - Use conceptual illustrations for introduction/motivation
        - When uncertain, err toward preserving visuals for more appropriate sections"""
    
    def decide_visuals(self, module_name, summary_text, candidates, used_visuals, max_new_tokens=200):
        # 过滤已使用的视觉元素
        available_candidates = [
            (v, score) for v, score in candidates 
            if (v.get("image_path") or v.get("table_path")) not in used_visuals
        ]
        if not available_candidates:
            return []

        candidate_text = ""
        for i, (v, score) in enumerate(available_candidates, 1):
            vtype = "Table" if "table_path" in v else "Figure"
            cap = v.get("caption", "")
            extra = v.get("table_text", "")[:300] if "table_path" in v else ""
            candidate_text += f"\n[{i}] ({vtype}, sim={score:.3f}) {cap}\n{extra}\n"

        prompt = f"""
You are a strategic visual resource allocator for a research paper webpage design.

CRITICAL ALLOCATION RULES:
1. STRICT ONE-TIME USE: Each visual can only be used ONCE in the entire project
2. SEQUENTIAL PRESERVATION: Never use visuals that appear later in the paper for earlier sections
3. TYPE-TO-SECTION MATCHING: Allocate visual types to their most appropriate sections

Current Module: {module_name}
Module Summary: {summary_text}

Available Candidate Visuals (NOT used in other modules):
{candidate_text}

**VISUAL TYPE TO SECTION MAPPING - STRICT GUIDELINES:**

**EXPERIMENTAL VISUALS → Reserve for Experiments/Results Sections:**
- Results tables, performance comparisons, ablation studies
- Quantitative evaluation charts, statistical analyses
- Benchmark comparison figures, accuracy/loss curves
- DO NOT use these in Introduction/Motivation sections

**METHODOLOGY VISUALS → Reserve for Methodology Sections:**
- Model architecture diagrams, system overviews
- Process flowcharts, algorithm pseudocode illustrations
- Technical component diagrams, framework schematics
- Parameter tables, configuration specifications

**CONCEPTUAL VISUALS → Use in Introduction/Motivation/Innovation Sections:**
- Problem illustrations, motivation diagrams
- Conceptual frameworks, high-level overviews
- Comparative analysis with prior work
- Domain-specific illustrative examples

**STRATEGIC SELECTION CRITERIA:**

1. **RELEVANCE ASSESSMENT:**
   - Does this visual directly support the core message of {module_name}?
   - Is there a stronger alignment with another module based on visual content?

2. **TYPE-SECTION ALIGNMENT:**
   - Experimental results → Experiments sections ONLY
   - Technical diagrams → Methodology sections ONLY  
   - Conceptual figures → Introduction/Motivation/Innovation sections ONLY

3. **IMPACT PRESERVATION:**
   - Save high-impact experimental visuals for results demonstration
   - Reserve technical architecture diagrams for methodology explanation
   - Keep conceptual illustrations for problem motivation/innovation

4. **CONSERVATIVE ALLOCATION:**
   - Select 0-2 visuals ONLY if they provide exceptional value
   - When in doubt, preserve the visual for potentially better-suited modules
   - Prioritize cross-module resource optimization over individual module completeness

**MODULE-SPECIFIC GUIDANCE FOR {module_name}:**

{self._get_module_specific_guidance(module_name)}

Return ONLY a JSON list of indices, e.g., [1,3] or [] if none are highly suitable.
Be extremely selective to preserve the most appropriate visuals for their ideal sections.
"""

        resp = self.llm_model.generate(query=prompt)

        try:
            matched = json.loads(re.search(r"\[.*?\]", resp, re.S).group())
        except Exception:
            matched = []
        return [available_candidates[i - 1][0] for i in matched if 1 <= i <= len(available_candidates)]



class SectionContentAgent:
    """章节内容生成代理，使用LLM2进行分层检索和内容生成"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
        
    def retrieve_relevant_titles(self, section_name, paper_sections, top_k=5):
        """检索与section_name相关的章节标题 - 增强版本"""
        
        # 创建section_name的同义词映射
        synonym_map = {
            "experiments": ["experiments", "experimental", "evaluation", "results", 
                        "performance", "benchmark", "analysis", "validation"],
            "methodology": ["methodology", "method", "approach", "technical", 
                        "framework", "architecture", "system"],
            "innovation": ["innovation", "contribution", "novelty", "technical_contribution"],
            "motivation": ["motivation", "introduction", "background", "problem"],
            "abstract": ["abstract", "summary", "overview"],
        }
        
        # 获取目标部分的所有可能关键词
        target_keywords = synonym_map.get(section_name.lower(), [section_name.lower()])
        
        # 先尝试使用LLM检索
        prompt = f"""Based on the given section name and paper structure, retrieve the most relevant section titles.
        Section name: {section_name}
        Possible keywords: {', '.join(target_keywords)}
        
        List of paper sections:
        {self._format_sections_list(paper_sections)}
        
        Please analyze and select the top {top_k} most relevant section titles from the list.
        Return only the list of section titles in order, with no explanations.
        Relevant section titles:"""
        
        try:
            response = self.llm_model.generate(query=prompt)
            titles = self._parse_titles_from_response(response, paper_sections)
                        
            return titles[:top_k]
        except Exception as e:
            print(f"❌ 检索相关标题失败: {e}")
            return self._keyword_based_retrieval(target_keywords, paper_sections, top_k)
    
    def generate_section_content(self, section_name, relevant_sections, previously_generated_content=None):
        """基于检索到的相关章节生成内容，考虑已有内容避免重复"""
        
        # 构建上下文信息，避免重复
        context_info = ""
        if previously_generated_content:
            context_info = f"""
**ALREADY COVERED IN OTHER SECTIONS (DO NOT REPEAT):**
{self._summarize_previous_content(previously_generated_content)}

**CRITICAL: Ensure this section introduces NEW information not covered above.**
"""
        
        prompt = f"""
You are an academic paper homepage assistant. Generate exceptionally clear, concise, and coherent content for the "{section_name}" section that flows naturally within the overall research narrative.

**COHERENCE AND UNIQUENESS REQUIREMENTS:**
- Create content that logically connects to the broader research story
- Introduce NEW information not covered in other sections
- Build upon concepts introduced in previous sections naturally
- Avoid repeating facts, examples, or explanations from other sections
- Ensure smooth conceptual flow between ideas

{context_info}

**Source Content:**
{self._format_relevant_sections(relevant_sections)}

**Narrative Flow Guidelines:**
• Start with content that naturally follows from previous sections
• Introduce concepts in logical sequence (general→specific, problem→solution)
• Use transitional language to connect ideas within the section
• Each paragraph should advance the section's unique contribution
• Maintain consistent terminology and conceptual framework

**Content Generation Strategy:**
• Extract the MOST ESSENTIAL information unique to this section
• Focus on this section's specific role in the research narrative
• Use connecting phrases to show relationship to broader context
• Apply <strong> only to 2-3 most important NEW concepts
• Eliminate any information redundant with other sections
• Ensure each sentence adds new value to the reader

**Section-Specific Focus:**
{self._get_section_focus_guidance(section_name)}

**Output Specifications:**
- Begin with content that naturally continues the research story
- No section titles, headings, or repetitive introductory phrases
- Ensure conceptual continuity with overall paper narrative
- Maintain consistent academic tone and terminology
- Keep length appropriate (typically 2-4 well-connected paragraphs)
- Verify NO overlap with content from other sections

Generate the pure content body for the {section_name} section:
"""
        
        try:
            response = self.llm_model.generate(query=prompt)
            return self._clean_generated_content(response)
        except Exception as e:
            print(f"❌ 生成章节内容失败: {e}")
            return self._fallback_content_generation(section_name, relevant_sections)
    
    def _summarize_previous_content(self, previous_content):
        """总结已生成的内容，帮助避免重复"""
        summary = []
        for section_name, content in previous_content.items():
            # 提取关键句子（前2句）
            sentences = re.split(r'[.!?]', content)
            key_sentences = [s.strip() for s in sentences[:2] if s.strip()]
            if key_sentences:
                summary.append(f"- {section_name}: {' '.join(key_sentences)}")
        
        return "\n".join(summary) if summary else "No previous content generated yet."

    def _get_section_focus_guidance(self, section_name):
        """为不同章节提供具体的内容聚焦指导"""
        focus_guidance = {
            "abstract": "Focus on overall contribution and significance - avoid detailed methodology or results",
            "motivation": "Emphasize problem importance and research gap - don't repeat introduction content",
            "innovation": "Highlight technical novelty and key innovations - avoid repeating methodology details",
            "methodology": "Explain core approach and technical framework - save implementation details for experiments",
            "experiments": "Focus on experimental setup and key findings - don't re-explain methodology"
        }
        
        section_lower = section_name.lower()
        for key, guidance in focus_guidance.items():
            if key in section_lower:
                return guidance
        
        return "Focus on this section's unique contribution to the overall research narrative."
    
    def _format_sections_list(self, paper_sections):
        """格式化章节列表用于提示"""
        sections_list = []
        for i, section in enumerate(paper_sections):
            title = section.get("title", "").strip()
            if title:
                sections_list.append(f"{i+1}.\n")
        return "\n".join(sections_list)
    
    def _parse_titles_from_response(self, response, paper_sections):
        """从模型响应中解析章节标题"""
        titles = []
        
        # 尝试多种解析方式
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配编号格式: "1. 标题" 或 "- 标题"
            match = re.match(r'^(\d+\.\s*|-\s*)(.+)', line)
            if match:
                potential_title = match.group(2).strip()
                # 在论文章节中查找匹配的标题
                for section in paper_sections:
                    section_title = section.get("title", "").strip()
                    if section_title and self._titles_match(section_title, potential_title):
                        titles.append(section_title)
                        break
        
        # 如果解析失败，使用备用方法
        if not titles:
            titles = self._fallback_title_matching(response, paper_sections)
            
        return list(set(titles))  # 去重
    
    def _titles_match(self, actual_title, extracted_title):
        """判断两个标题是否匹配"""
        actual_clean = re.sub(r'[^\w\s]', '', actual_title.lower())
        extracted_clean = re.sub(r'[^\w\s]', '', extracted_title.lower())
        
        # 完全匹配或包含关系
        return (actual_clean == extracted_clean or 
                actual_clean in extracted_clean or 
                extracted_clean in actual_clean)
    
    def _fallback_title_matching(self, response, paper_sections):
        """备用标题匹配方法"""
        titles = []
        response_lower = response.lower()
        
        for section in paper_sections:
            section_title = section.get("title", "").strip()
            if section_title:
                # 简单的关键词匹配
                title_keywords = set(re.findall(r'\b\w+\b', section_title.lower()))
                response_keywords = set(re.findall(r'\b\w+\b', response_lower))
                
                common_keywords = title_keywords.intersection(response_keywords)
                if len(common_keywords) >= 2:  # 至少有2个共同关键词
                    titles.append(section_title)
        
        return titles[:5]  # 最多返回5个
    
    def _format_relevant_sections(self, relevant_sections):
        """格式化相关章节内容"""
        formatted = []
        for section in relevant_sections:
            title = section.get("title", "Untitled")
            content = section.get("content", "").strip()
            if content:
                formatted.append(f"【{title}】\n{content}\n")
        return "\n".join(formatted)
    
    def _clean_generated_content(self, content):
        """清理生成的内容"""
        # 移除可能的提示词残留
        content = re.sub(r'^(请生成|生成内容|内容:|#+)\s*', '', content, flags=re.IGNORECASE)
        content = content.strip()
        
        # 确保以完整的句子结束
        if content and not content.endswith(('.', '。')):
            content += '.'
            
        return content
    
    def _keyword_based_retrieval(self, target_keywords, paper_sections, top_k):
        """基于关键词的检索方法"""
        relevant_titles = []
        
        for section in paper_sections:
            title = section.get("title", "").lower()
            for keyword in target_keywords:
                if keyword in title:
                    relevant_titles.append(section.get("title", ""))
                    break
        
        return relevant_titles[:top_k]
    
    def _fallback_content_generation(self, section_name, relevant_sections):
        """备用内容生成方法"""
        if not relevant_sections:
            return f"Content for {section_name} not available."
        
        # 简单拼接相关内容
        contents = []
        for section in relevant_sections:
            content = section.get("content", "").strip()
            if content:
                contents.append(content)
        
        if contents:
            # 取第一个相关内容作为主要内容
            main_content = contents[0]
            # 简单截断以避免过长
            if len(main_content) > 500:
                sentences = re.split(r'[.!?。！？]', main_content)
                summary = []
                total_length = 0
                for sentence in sentences:
                    if sentence.strip():
                        summary.append(sentence.strip())
                        total_length += len(sentence)
                        if total_length > 300:
                            break
                return '. '.join(summary) + '.'
            return main_content
        else:
            return f"Content for {section_name} not available."

# ==========================================================
# 🔹 主类：简化版本
# ==========================================================
class PaperHomepageGenerator:
    def __init__(self, content_json_path, modules_json_path, template_path,
                 qwen_model, bge_path, images_json_path=None, tables_json_path=None, csv_path=None):
        self.content_json_path = content_json_path
        self.modules_json_path = modules_json_path
        self.images_json_path = images_json_path
        self.tables_json_path = tables_json_path
        self.template_path = template_path
        self.qwen_model = qwen_model  # 现在直接传入LLM2实例
        self.bge_path = bge_path
        self.csv_path = csv_path

        self.paper_content = self._load_json(self.content_json_path)
        self.modules_content = self._load_json(self.modules_json_path)
        self.images_data = self._load_json(self.images_json_path) or {}
        self.tables_data = self._load_json(self.tables_json_path) or {}
        self.link_data = self._load_csv_links()

        self.qwen = QwenDecider(self.qwen_model)  # 传入LLM2实例
        self.bge = BGEEmbedder(self.bge_path)
        
        # 新增代理
        self.planner_agent = PlannerAgent(self.qwen_model)
        self.section_agent = SectionContentAgent(self.qwen_model)
        
        self.used_visuals = set()
        self.formatter = TextFormatter()
        self.table_counter = 0
        
        self.output_assets_dir = None
        self.assets_mapping = {}
        self.planned_sections = []  # 存储规划的部分

    def _load_csv_links(self):
        """从CSV文件加载论文链接信息"""
        if not self.csv_path or not os.path.exists(self.csv_path):
            print(f"⚠️ CSV文件不存在: {self.csv_path}")
            return {}
        
        try:
            df = pd.read_csv(self.csv_path)
            link_dict = {}
            
            for _, row in df.iterrows():
                title = row.get('title', '').strip()
                if title:
                    link_dict[title] = {
                        'paper_url': row.get('paper_url', '#'),
                        'homepage': row.get('homepage', '#')
                    }
            
            print(f"✅ 从CSV加载了 {len(link_dict)} 篇论文的链接信息")
            return link_dict
            
        except Exception as e:
            print(f"❌ 加载CSV链接失败: {e}")
            return {}

    def _load_json(self, path):
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_output_directory(self, output_path):
        """创建输出目录结构"""
        output_dir = Path(output_path).parent
        html_filename = Path(output_path).name
        
        if '.' in html_filename:
            folder_name = html_filename.rsplit('.', 1)[0]
            output_folder = output_dir / folder_name
        else:
            output_folder = output_dir / html_filename
            
        output_folder.mkdir(parents=True, exist_ok=True)
        self.output_assets_dir = output_folder / "assets"
        self.output_assets_dir.mkdir(exist_ok=True)
        
        return output_folder / "index.html"

    def _copy_asset(self, original_path):
        """复制资源文件到输出目录并返回相对路径"""
        if not original_path or not os.path.exists(original_path):
            return original_path
            
        if original_path in self.assets_mapping:
            return self.assets_mapping[original_path]
        
        original_file = Path(original_path)
        new_filename = f"asset_{len(self.assets_mapping)}_{original_file.name}"
        relative_path = f"assets/{new_filename}"
        
        destination = self.output_assets_dir / new_filename
        try:
            shutil.copy2(original_path, destination)
            self.assets_mapping[original_path] = relative_path
            print(f"✅ 复制资源: {original_path} -> {relative_path}")
            return relative_path
        except Exception as e:
            print(f"❌ 复制资源失败: {original_path}, 错误: {e}")
            return original_path

    def _extract_basic_paper_info(self):
        """提取论文基本信息（标题、作者、链接）"""
        # 提取标题
        title = self.paper_content.get("sections", [{}])[0].get("title", "Untitled Paper")
        
        # 提取作者信息
        first_content = self.paper_content.get("sections", [{}])[0].get("content", "")
        authors, affiliations, project_link = self._extract_authors_from_content(first_content)
        authors = authors or ["Anonymous"]
        
        # 从CSV获取链接
        csv_links = self._get_links_for_paper(title)
        paper_url = csv_links.get('paper_url', '#')
        homepage_url = csv_links.get('homepage', '#')
        
        # 尝试提取发表信息
        publication_info = self._extract_publication_info()
        
        return {
            "title": title,
            "authors": ', '.join(authors),
            "affiliations": ', '.join(affiliations),
            "publication_info": publication_info,
            "year": "2025",  # 默认年份
            "links": {
                "paper": paper_url,
                "project_page": homepage_url
            }
        }

    def _extract_publication_info(self):
        """尝试从论文内容中提取发表信息"""
        # 检查前几个章节
        for i, section in enumerate(self.paper_content.get("sections", [])[:5]):
            content = section.get("content", "")
            # 查找可能的会议/期刊信息
            patterns = [
                r'\b(arXiv|CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|ACL|EMNLP|NAACL)\b',
                r'\b(Proceedings of|Conference on|Workshop on)\b',
                r'\b\d{4}\b.*\b(Conference|Symposium|Workshop)\b'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # 返回第一匹配
                    return matches[0] if isinstance(matches[0], str) else ' '.join(matches[0])
        
        # 默认返回
        return "Conference on AI Research"

    def _extract_authors_from_content(self, text):
        # 保持原有的作者提取逻辑
        clean_text = re.sub(r'[\*/a-zA-Z]*a0', '', text)
        clean_text = re.sub(r'[\*∗]', '', text)

        lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
        authors_line, aff_line, link = "", "", "#"

        for l in lines:
            if "university" in l.lower() or "institute" in l.lower() or "school" in l.lower() or "laboratory" in l.lower():
                aff_line += " " + l
            elif "http" in l.lower() or "https" in l.lower():
                m = re.search(r'(https?://[^\s]+)', l)
                if m:
                    link = m.group(1)
            else:
                authors_line += " " + l

        author_pattern = r'(\d+)\s*([A-Z][a-zA-Z\-\.]+\s+[A-Z][a-zA-Z\-\.]+)'
        authors_with_ids = re.findall(author_pattern, authors_line)

        if authors_with_ids:
            authors = [name.strip() for _, name in authors_with_ids]
        else:
            authors = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-zA-Z\-]+)\b', authors_line)
            authors = [a for a in authors if len(a.split()) == 2 and not any(x.lower() in a.lower() for x in ["University", "Laboratory"])]

        aff_pattern = r'(\d+)\s+([^0-9\n]+)'
        aff_matches = re.findall(aff_pattern, aff_line)
        aff_dict = {num: aff.strip() for num, aff in aff_matches}
        if not aff_dict and aff_line:
            aff_dict = {"1": aff_line.strip()}

        affiliations = list(aff_dict.values()) or []

        return authors, affiliations, link

    def _extract_section_content(self, section_name, previously_generated=None):
        """使用新的Agent根据部分名称提取内容，考虑已有内容避免重复"""
        # 特殊处理的章节（保持原有逻辑）
        section_name_lower = section_name.lower()
        
        # 首先检查是否有直接的模块内容
        for module_name in self.modules_content.keys():
            if module_name.lower() == section_name_lower or section_name_lower in module_name.lower():
                content = self.modules_content[module_name].get("summary", "")
                print(f"✅ 从模块获取内容: {module_name}")
                return self.formatter.format_text(content)
        
        # 处理特殊章节
        if section_name_lower == 'abstract':
            for sec in self.paper_content.get("sections", []):
                if "abstract" in sec.get("title", "").lower():
                    return self.formatter.format_text(sec.get("content", ""))
        
        # 使用新的Agent进行分层检索和生成
        print(f"🔍 使用Agent检索和生成内容: {section_name}")
        
        # 第一步：检索相关标题
        paper_sections = self.paper_content.get("sections", [])
        relevant_titles = self.section_agent.retrieve_relevant_titles(section_name, paper_sections)
        
        print(f"📚 检索到相关章节: {relevant_titles}")
        
        # 第二步：获取相关章节的内容
        relevant_sections = []
        for title in relevant_titles:
            for section in paper_sections:
                if section.get("title", "").strip() == title:
                    relevant_sections.append(section)
                    break
        
        # 第三步：生成内容，传入已生成内容以避免重复
        if relevant_sections:
            generated_content = self.section_agent.generate_section_content(
                section_name, relevant_sections, previously_generated
            )
            formatted_content = self.formatter.format_text(generated_content)
            print(f"✅ 生成内容完成，长度: {len(formatted_content)}")
            return formatted_content
        else:
            print(f"⚠️ 未找到相关内容，使用备用方法")
            # 尝试在论文章节中直接查找
            for sec in paper_sections:
                if section_name_lower in sec.get("title", "").lower():
                    return self.formatter.format_text(sec.get("content", ""))
            
            return f"Content for {section_name} not available."

    def _select_visuals_for_section(self, section_name, section_content):
        """为特定部分选择视觉元素"""
        candidates = []
        for fig in self.images_data.values():
            # 使用原始标题进行相似度计算，因为原始标题可能包含更多信息
            candidates.append({"text": fig.get("caption", ""), **fig})
        for tb in self.tables_data.values():
            text = tb.get("caption", "") + "\n" + tb.get("table_text", "")
            candidates.append({"text": text, **tb})

        if not candidates:
            return []

        top_candidates = self.bge.similarity(section_content, candidates, top_k=6)
        selected_visuals = self.qwen.decide_visuals(
            section_name, section_content, top_candidates, self.used_visuals
        )

        for v in selected_visuals:
            path = v.get("image_path") or v.get("table_path")
            self.used_visuals.add(path)

        return selected_visuals[:2]

    def _render_visual_html(self, v):
        """渲染视觉元素 - 简化版本，表格直接作为图片"""
        if "table_path" in v and os.path.exists(v["table_path"]):
            # 表格直接作为图片处理
            self.table_counter += 1
            table_id = self.table_counter
            # 使用清理后的标题
            caption = self.formatter.clean_caption(v.get("caption", ""))
            print(f"📊 直接使用表格图片: {v.get('table_path')}")
            return self._render_table_as_image(v, caption, table_id)

        if "image_path" in v or "table_path" in v:
            relative_path = self._copy_asset(v.get("image_path") or v.get("table_path"))
            # 使用清理后的标题
            caption = self.formatter.clean_caption(v.get("caption", ""))
            
            width = v.get('width', 0)
            height = v.get('height', 0)
            
            return f"""
            <div class="visual my-6 text-center">
                <div class="max-w-3xl mx-auto">
                    <img src="{relative_path}" alt="{caption}" 
                        class="w-full h-auto rounded-lg shadow-md mx-auto"
                        loading="lazy" decoding="async">
                </div>
                <p class="text-sm italic text-gray-600 mt-2">
                    {caption}
                </p>
            </div>
            """
        
        print(f"⚠️ 不支持的视觉元素类型: {v.keys()}")
        return ""

    def _render_table_as_image(self, v, caption, table_id):
        """将表格作为图片渲染"""
        table_img_path = v.get("table_path")
        if table_img_path and os.path.exists(table_img_path):
            relative_path = self._copy_asset(table_img_path)
            print(f"✅ 表格作为图片显示: {table_img_path}")
            return f"""
            <div class="table-visualization my-8 bg-white rounded-xl shadow-lg overflow-hidden">
                <div class="p-6">
                    <div class="overflow-x-auto rounded-lg border border-gray-200">
                        <img src="{relative_path}" alt="{caption}" class="w-full h-auto rounded-lg">
                    </div>
                    <p class="text-sm italic text-gray-600 mt-4 text-center">
                        {caption}
                    </p>
                </div>
            </div>
            """
        else:
            return f"""
            <div class="visual my-6 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p class="text-sm text-yellow-800">⚠️ 表格图片不存在</p>
                <p class="text-sm italic text-gray-600 mt-2 text-center">
                    {caption}
                </p>
            </div>
            """

    def _protect_citations(self, text):
        """保护引用标记"""
        protected_text = re.sub(r'\[([^\[\]]+)\]', r'<span class="no-mathjax">[\1]</span>', text)
        return protected_text

    def _build_paper_info(self, planned_sections):
        """构建论文信息，根据规划的部分动态生成内容"""
        # 提取基本信息
        basic_info = self._extract_basic_paper_info()
        title = basic_info["title"]
        authors = basic_info["authors"]
        affiliations = basic_info["affiliations"]
        links = basic_info["links"]
        
        print(f"📊 规划的部分: {planned_sections}")
        
        # 为每个规划的部分生成内容，按顺序传递已生成内容
        section_contents = {}
        previously_generated = {}
        
        for section in planned_sections:
            print(f"\n📝 处理部分: '{section}'")
            
            # 生成内容
            content = self._extract_section_content(section, previously_generated)
            protected_content = self._protect_citations(content)
            
            # 存储已生成内容供后续章节参考
            previously_generated[section] = content
            
            # 为有内容的部分选择视觉元素
            visuals = []
            if content and len(content.strip()) > 50:
                visuals = self._select_visuals_for_section(section, content)
            
            # 构建该部分的HTML
            section_html = f"<div class='section-content'>\n"
            section_html += f"<p>{protected_content}</p>\n"
            
            for v in visuals:
                visual_html = self._render_visual_html(v)
                section_html += visual_html + "\n"
            
            section_html += "</div>"
            section_contents[section] = section_html

        # 生成BibTeX
        bibkey = re.sub(r"\W+", "", title)[:15]
        bibtex = f"""@inproceedings{{{bibkey}2025,
        title={{ {title} }},
        author={{ {" and ".join(authors.split(','))} }},
        booktitle={{ {basic_info['publication_info']} }},
        year={{2025}},
        }}"""
        
        # 构建链接数据
        links_data = {
            "paper": links.get("paper", "#"),
            "code": "#",  # 模板中需要的占位符
            "dataset": "#",  # 模板中需要的占位符
            "project_page": links.get("project_page", "#")
        }
        
        # 返回完整信息
        paper_info = {
            "title": title,
            "authors": authors,
            "affiliations": affiliations,
            "publication_info": basic_info['publication_info'],
            "year": basic_info['year'],
            "bibtex": bibtex,
            "links": links_data,
            "planned_sections": planned_sections,
        }
        
        # 添加各个部分的内容
        paper_info.update(section_contents)
        
        # 添加模板需要的变量
        paper_info["abstract"] = section_contents.get("abstract", "")
        paper_info["motivation"] = section_contents.get("motivation", "")
        paper_info["innovation"] = section_contents.get("innovation", "")
        paper_info["methodology"] = section_contents.get("methodology", "")
        paper_info["experiments"] = section_contents.get("experiments", "")
        
        return paper_info

    def _get_links_for_paper(self, paper_title):
        """根据论文标题获取链接"""
        if paper_title in self.link_data:
            return self.link_data[paper_title]
        
        normalized_title = re.sub(r'[^\w\s]', '', paper_title).lower().strip()
        for csv_title, links in self.link_data.items():
            normalized_csv_title = re.sub(r'[^\w\s]', '', csv_title).lower().strip()
            if normalized_title == normalized_csv_title:
                return links
        
        for csv_title, links in self.link_data.items():
            paper_keywords = ' '.join(paper_title.split()[:5]).lower()
            csv_keywords = ' '.join(csv_title.split()[:5]).lower()
            
            if paper_keywords in csv_keywords or csv_keywords in paper_keywords:
                print(f"🔍 部分匹配: '{paper_title}' -> '{csv_title}'")
                return links
        
        print(f"⚠️ 未在CSV中找到论文链接: {paper_title}")
        return {'paper_url': '#', 'homepage': '#'}
    
    def _add_mathjax_support(self, html):
        """添加MathJax支持（如果模板中没有）"""
        if 'MathJax' in html:
            print("✅ 模板已有MathJax支持")
            return html
        
        mathjax_script = """
        <!-- MathJax支持 -->
        <script>
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$']],
                    displayMath: [['$$', '$$']]
                },
                svg: {
                    fontCache: 'global'
                },
                options: {
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                    processEscapes: false
                }
            };
        </script>
        <script type="text/javascript" id="MathJax-script" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
        </script>
        
        <style>
            mjx-container {
                display: inline-block !important;
                line-height: normal !important;
            }
            .MathJax {
                font-style: normal !important;
                font-weight: normal !important;
            }
            .no-mathjax {
                font-style: normal !important;
                color: inherit !important;
                background-color: transparent !important;
                display: inline !important;
            }
        </style>
        """
        
        # 插入到head标签中
        head_pattern = re.compile(r'<head>', re.IGNORECASE)
        match = head_pattern.search(html)
        
        if match:
            end_pos = match.end()
            html = html[:end_pos] + mathjax_script + html[end_pos:]
            print("✅ 添加MathJax支持到head标签")
        else:
            # 如果找不到head标签，在html标签后添加
            html_pattern = re.compile(r'<html[^>]*>', re.IGNORECASE)
            html_match = html_pattern.search(html)
            if html_match:
                end_pos = html_match.end()
                html = html[:end_pos] + f'\n<head>\n{mathjax_script}\n</head>' + html[end_pos:]
                print("✅ 创建head标签并添加MathJax支持")
            else:
                html = f'<head>\n{mathjax_script}\n</head>\n' + html
                print("✅ 在HTML开头添加head标签和MathJax支持")
        
        return html
    
    def _copy_static_resources_simple(self, output_folder, template_dir):
        """简单复制静态资源，不修改CSS"""
        print(f"📁 复制模板资源: {template_dir} -> {output_folder}")
        
        # 复制所有非HTML文件
        for item in template_dir.iterdir():
            if item.is_file() and item.suffix not in ['.html', '.jinja', '.jinja2']:
                try:
                    shutil.copy2(item, output_folder / item.name)
                    print(f"✅ 复制文件: {item.name}")
                except Exception as e:
                    print(f"❌ 复制文件失败 {item.name}: {e}")
        
        # 复制子目录（除了已处理的assets）
        for item in template_dir.iterdir():
            if item.is_dir() and item.name != "assets" and not item.name.startswith('.'):
                dest_dir = output_folder / item.name
                try:
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(item, dest_dir)
                    print(f"✅ 复制目录: {item.name}")
                except Exception as e:
                    print(f"❌ 复制目录失败 {item.name}: {e}")
    
    def generate_homepage(self, output_path):
        """生成主页 - 简化版本"""
        # 设置输出目录
        final_html_path = self._setup_output_directory(output_path)
        self.assets_mapping = {}
        
        # 第一步：提取基本信息
        print("📋 第一步：提取论文基本信息...")
        basic_info = self._extract_basic_paper_info()
        print(f"✅ 提取基本信息: {basic_info['title']}")
        
        # 第二步：规划主页部分
        print("🎯 第二步：规划主页展示部分...")
        self.planned_sections = self.planner_agent.plan_homepage_sections(
            self.paper_content, self.modules_content
        )
        print(f"✅ 规划完成: {self.planned_sections}")
        
        # 第三步：构建完整论文信息
        print("📊 第三步：构建完整论文信息...")
        paper_info = self._build_paper_info(self.planned_sections)
        
        # 第四步：合并所有信息
        print("🔄 第四步：合并信息...")
        all_data = {
            **basic_info,
            **paper_info,
        }
        
        # 第五步：加载原始模板并渲染
        print("🎨 第五步：加载原始模板...")
        template_dir = Path(self.template_path).parent
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(Path(self.template_path).name)
        
        print("🚀 第六步：渲染最终HTML...")
        final_html = template.render(**all_data)
        
        # 添加MathJax支持（如果模板中没有）
        if 'MathJax' not in final_html:
            print("➕ 添加MathJax支持...")
            final_html = self._add_mathjax_support(final_html)
        
        # 保存HTML文件
        with open(final_html_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        
        # 第七步：复制静态资源
        print("📁 第七步：复制静态资源...")
        self._copy_static_resources_simple(final_html_path.parent, template_dir)
        
        print(f"✅ 主页生成完成: {final_html_path}")
        print(f"📁 资源文件保存在: {self.output_assets_dir}")
        print(f"📊 共复制了 {len(self.assets_mapping)} 个资源文件")
        print(f"🎯 规划展示部分: {self.planned_sections}")
        
        return final_html_path

# ==========================================================
# 🔹 批量生成模式
# ==========================================================

if __name__ == "__main__":
    # 初始化文本模型
    llm_model = LLM2('Qwen2.5-7B-Instruct')
    
    template_path_list = ['/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/muban_clean/bluepaper.html']
    out_dir_list = ['orangepaper']
    
    # CSV文件路径
    csv_path = "/home/gaojuanru/mnt_link/gaojuanru/twittergenerate/sample_papers_updated.csv"
    
    for template_path, out_dir in zip(template_path_list, out_dir_list):
        base_dir = "/home/gaojuanru/mnt_link/gaojuanru/twittergenerate/jiexi"
        bge_path = "/mnt/gaojuanru/twittergenerate/cache/huggingface/BAAI/bge-m3"

        # 遍历 base_dir 下的所有论文子文件夹
        for subdir in sorted(os.listdir(base_dir)):
            sub_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(sub_path):
                continue

            # 自动匹配四类 JSON 文件
            content_json = os.path.join(sub_path, f"{subdir}_content.json")
            modules_json = os.path.join(sub_path, f"{subdir}_content_modules.json")
            images_json = os.path.join(sub_path, f"{subdir}_images.json")
            tables_json = os.path.join(sub_path, f"{subdir}_tables.json")

            # 确保至少有 content 和 modules 文件
            if not (os.path.exists(content_json) and os.path.exists(modules_json)):
                print(f"⚠️ 缺少主要JSON文件: {subdir}, 跳过。")
                continue

            # 输出文件路径
            output_html = os.path.join(
                "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/2",
                f'{out_dir}',
                f"paper_homepage_{subdir}.html"
            )
            os.makedirs(os.path.dirname(output_html), exist_ok=True)

            print(f"\n🚀 正在为 {subdir} 生成主页...")

            try:
                generator = PaperHomepageGenerator(
                    content_json_path=content_json,
                    modules_json_path=modules_json,
                    images_json_path=images_json if os.path.exists(images_json) else None,
                    tables_json_path=tables_json if os.path.exists(tables_json) else None,
                    template_path=template_path,
                    qwen_model=llm_model,
                    bge_path=bge_path,
                    csv_path=csv_path
                )

                generator.generate_homepage(output_html)
                print(f"✅ {subdir} 主页生成成功")
                
            except Exception as e:
                print(f"❌ {subdir} 生成失败: {e}")

        print("\n🎉 所有可用论文主页生成完成!")