# PaperPageAI：论文主页与讲解自动生成工具

## 项目简介
PaperPageAI 是一套基于 RAG（检索增强生成）与大语言模型的自动化工具链，支持从 PDF 论文直接生成结构化学术主页（HTML 格式）和论文推文（Markdown 格式）。工具链整合了 PDF 解析、向量检索、内容生成、视觉资源匹配等核心功能，无需人工干预即可产出符合学术规范的展示内容。

## 核心功能
- **PDF 智能解析**：提取论文章节、图表（图片+表格）、作者信息，自动过滤参考文献后冗余内容
- **双模式输出**：
  1. 学术主页（HTML）：包含 Abstract、Motivation、Innovation、Methodology、Experiments 五大核心模块
  2. 推文（Markdown）：结构化呈现论文核心内容，适配学术分享场景

## 环境依赖
1. Python 版本：3.10
2. 依赖安装：
```bash
pip install -r requirements.txt
```
3. 模型依赖（自动缓存，需联网）：
   - BGE-M3（向量检索模型）
   - Qwen2.5-7B-Instruct（大语言模型）

## 运行顺序（核心流程）
### 单文件运行（推荐，通过交互界面）
1. 启动主程序：
```bash
python output.py
```
2. 交互选择功能：
   - 输入 `1`：生成论文主页（需指定 PDF 路径、模板、输出目录）
   - 输入 `2`：生成论文推文（需指定 PDF 路径、输出目录）
   - 输入 `3`：退出程序
3. 程序自动执行完整流程，无需手动干预

### 手动分步运行（进阶使用）
1. **PDF 解析**：将 PDF 转为结构化 JSON（章节+图表）
```bash
python parse.py
```
- 输入：PDF 文件（默认路径：`/PaperPageAI/pdf`）
- 输出：`{论文名}_content.json`（章节内容）、`{论文名}_images.json`（图片信息）、`{论文名}_tables.json`（表格信息）

2. **模块内容生成**：基于 BGE 检索+Qwen 生成核心模块文本
```bash
python bge_search.py
```
- 输入：解析后的 `_content.json` 文件
- 输出：`{论文名}_content_modules.json`（五大模块结构化文本）

3. **最终成果生成**：
   - 生成主页：`python genhtml.py`（输入 JSON 文件+模板，输出 HTML 主页）
   - 生成推文：`python tuiwen.py`（输入 JSON 文件，输出 Markdown 讲解）

### 批量运行
1. 批量生成主页：直接运行 `genhtml.py`（自动遍历 `jiexi` 目录下所有论文文件夹）
2. 批量生成讲解：直接运行 `tuiwen.py`（自动遍历 `jiexi` 目录下所有论文文件夹）

## 目录结构说明
```
PaperPageAI/
├── pdf/                # 输入PDF文件存放目录
├── jiexi/              # 解析后JSON文件输出目录（自动创建）
├── muban_clean/        # HTML主页模板目录（内置多个风格模板）
├── html_output/        # 主页输出目录（自动创建）
├── tweet_output/       # 讲解输出目录（自动创建）
├── parse.py            # PDF解析脚本
├── bge_search.py       # 模块内容生成脚本
├── genhtml.py          # 主页生成脚本
├── tuiwen.py           # 推文生成脚本
├── pipeline.py         # 流水线核心脚本
├── output.py           # 交互入口脚本
└── requirements.txt    # 依赖清单
```
