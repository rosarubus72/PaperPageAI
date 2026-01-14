import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import subprocess
import sys

class PaperHomepagePipeline:
    """论文主页生成流水线 - 封装整个流程"""
    
    def __init__(
        self,
        template_dir: str = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/muban_clean",
        bge_model_path: str = "/home/gaojuanru/.cache/huggingface/BAAI/bge-m3",
        qwen_model_path: str = "/home/gaojuanru/.cache/huggingface/Qwen/Qwen2.5-7B-Instruct",
        csv_path: str = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/sample_papers_updated.csv",
        llm2_model: str = "Qwen2.5-7B-Instruct"
    ):
        """
        初始化流水线
        
        Args:
            template_dir: 模板目录路径
            bge_model_path: BGE模型路径
            qwen_model_path: Qwen模型路径
            csv_path: 论文链接CSV文件路径
            llm2_model: LLM2模型名称
        """
        self.template_dir = Path(template_dir)
        self.bge_model_path = bge_model_path
        self.qwen_model_path = qwen_model_path
        self.csv_path = csv_path
        self.llm2_model = llm2_model
        
        # 验证路径
        if not self.template_dir.exists():
            raise ValueError(f"模板目录不存在: {template_dir}")
        if not Path(csv_path).exists():
            print(f"⚠️ CSV文件不存在: {csv_path}")
        
        print(f"✅ 流水线初始化完成")
        print(f"   - 模板目录: {self.template_dir}")
        print(f"   - BGE模型: {self.bge_model_path}")
        print(f"   - Qwen模型: {self.qwen_model_path}")
    
    def run_full_pipeline(
        self,
        pdf_path: str,
        output_type: str = "homepage",  # "homepage" 或 "explanation"
        template_name: str = "orangepaper.html",
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        运行完整流水线
        
        Args:
            pdf_path: PDF文件路径
            output_type: 输出类型 - "homepage" 或 "explanation"
            template_name: 模板文件名（仅主页生成需要）
            output_dir: 输出目录（默认为PDF同目录）
            
        Returns:
            Dict: 包含生成结果的信息
        """
        print(f"\n🚀 开始处理PDF: {pdf_path}")
        print(f"📝 输出类型: {output_type}")
        
        if output_type not in ["homepage", "explanation"]:
            raise ValueError(f"不支持的输出类型: {output_type}。请使用 'homepage' 或 'explanation'")
        
        # 1. 创建临时工作目录
        temp_dir = Path(tempfile.mkdtemp(prefix="paper_homepage_"))
        print(f"📁 创建临时目录: {temp_dir}")
        
        # 2. 验证PDF文件
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        try:
            # 第一步：解析PDF
            print("\n📄 第一步：解析PDF...")
            parse_output = self._run_parse_step(pdf_path, temp_dir)
            
            # 第二步：BGE搜索生成模块
            print("\n🔍 第二步：BGE搜索生成模块...")
            bge_output = self._run_bge_search_step(parse_output["content_json"], temp_dir)
            
            # 第三步：根据输出类型选择生成流程
            if output_type == "homepage":
                print("\n🏠 第三步：生成论文主页...")
                return self._run_homepage_generation_step(
                    parse_output=parse_output,
                    bge_output=bge_output,
                    template_name=template_name,
                    output_dir=output_dir,
                    paper_name=pdf_path_obj.stem
                )
            else:  # explanation
                print("\n🐦 第三步：生成论文讲解推文...")
                return self._run_explanation_generation_step(
                    parse_output=parse_output,
                    bge_output=bge_output,
                    output_dir=output_dir,
                    paper_name=pdf_path_obj.stem
                )
            
        except Exception as e:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            print(f"❌ 流水线执行失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pdf_path": pdf_path
            }
        finally:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                print(f"🧹 清理临时目录: {temp_dir}")
    
    def _run_parse_step(self, pdf_path: str, output_dir: Path) -> Dict:
        """运行PDF解析步骤"""
        # 导入解析模块
        sys.path.append(str(Path(__file__).parent))
        
        # 使用原始的1parse.py逻辑
        from parse import process_pdf
        
        # 运行解析
        pdf_name = Path(pdf_path).stem
        paper_output_dir = output_dir / pdf_name
        paper_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用原解析函数
        process_pdf(pdf_path, str(output_dir))
        
        # 收集输出文件
        output_files = {
            "content_json": paper_output_dir / f"{pdf_name}_content.json",
            "images_json": paper_output_dir / f"{pdf_name}_images.json",
            "tables_json": paper_output_dir / f"{pdf_name}_tables.json",
            "images_dir": paper_output_dir / "images_and_tables"
        }
        
        # 验证文件
        if not output_files["content_json"].exists():
            raise FileNotFoundError(f"内容JSON未生成: {output_files['content_json']}")
        
        print(f"✅ PDF解析完成:")
        print(f"   - 内容JSON: {output_files['content_json']}")
        print(f"   - 图片JSON: {output_files['images_json'] if output_files['images_json'].exists() else '未生成'}")
        print(f"   - 表格JSON: {output_files['tables_json'] if output_files['tables_json'].exists() else '未生成'}")
        
        return {
            "output_dir": str(paper_output_dir),
            "content_json": str(output_files["content_json"]),
            "images_json": str(output_files["images_json"]) if output_files["images_json"].exists() else None,
            "tables_json": str(output_files["tables_json"]) if output_files["tables_json"].exists() else None,
            "paper_name": pdf_name
        }
    
    def _run_bge_search_step(self, content_json_path: str, output_dir: Path) -> Dict:
        """运行BGE搜索步骤"""
        # 导入BGE搜索模块
        sys.path.append(str(Path(__file__).parent))
        
        # 使用原始的3bge_search.py逻辑，但修改为单文件处理
        from bge_search import BGEInternalRetriever, QwenSummarizer
        
        # 初始化模型
        retriever = BGEInternalRetriever(self.bge_model_path)
        summarizer = QwenSummarizer(self.qwen_model_path)
        
        # 加载论文
        retriever.load_document(content_json_path)
        
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
                "Describe the core architecture, main modules, and the process by which it solves the problem."
            ),
            "Experiments": (
                "What experiments were conducted and what are the key findings? "
                "Summarize how the results demonstrate the effectiveness or advantages of the method."
            )
        }
        
        summaries = {}
        paper_name = Path(content_json_path).stem.replace("_content", "")
        
        print(f"📚 为论文 '{paper_name}' 生成模块...")
        
        for module_name, query in queries.items():
            print(f"  🔍 处理模块: {module_name}")
            results = retriever.retrieve(query, top_k=5)
            summary = summarizer.generate_summary(module_name, query, results)
            
            summaries[module_name] = {
                "query": query,
                "summary": summary,
                "retrieved_sections": results
            }
        
        # 保存模块JSON
        output_file = Path(content_json_path).parent / f"{paper_name}_content_modules.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        
        print(f"✅ BGE搜索完成: {output_file}")
        
        return {
            "output_file": str(output_file),
            "modules": summaries
        }
    
    def _run_homepage_generation_step(
        self,
        parse_output: Dict,
        bge_output: Dict,
        template_name: str,
        output_dir: Path,
        paper_name: str
    ) -> Dict:
        """运行主页生成步骤"""
        # 验证模板文件
        template_path = self.template_dir / template_name
        if not template_path.exists():
            # 尝试查找可用模板
            available_templates = list(self.template_dir.glob("*.html"))
            if available_templates:
                template_path = available_templates[0]
                print(f"⚠️ 指定模板不存在，使用默认模板: {template_path.name}")
            else:
                raise FileNotFoundError(f"模板目录中没有HTML模板文件")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = Path(parse_output["output_dir"]).parent / "homepage_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导入生成模块
        sys.path.append(str(Path(__file__).parent))
        
        # 需要导入LLM2类
        from llms.llm2 import LLM2
        
        # 初始化LLM2
        print(f"🤖 初始化LLM2模型: {self.llm2_model}")
        llm_model = LLM2(self.llm2_model)
        
        # 导入主页生成器
        from genhtml import PaperHomepageGenerator
        
        # 构建文件路径
        content_json = parse_output["content_json"]
        modules_json = bge_output["output_file"]
        images_json = parse_output["images_json"]
        tables_json = parse_output["tables_json"]
        
        # 创建生成器实例
        generator = PaperHomepageGenerator(
            content_json_path=content_json,
            modules_json_path=modules_json,
            template_path=str(template_path),
            qwen_model=llm_model,
            bge_path=self.bge_model_path,
            images_json_path=images_json,
            tables_json_path=tables_json,
            csv_path=self.csv_path
        )
        
        # 生成主页
        output_html_path = output_dir / f"{paper_name}_homepage.html"
        
        print(f"🎨 生成主页: {output_html_path}")
        generator.generate_homepage(str(output_html_path))
        
        # 获取规划的部分（如果可用）
        planned_sections = getattr(generator, 'planned_sections', [])
        
        # 找到实际的输出文件（生成器可能会创建子目录）
        final_html_path = Path(generator.output_assets_dir).parent / "index.html" if hasattr(generator, 'output_assets_dir') else output_html_path
        
        result = {
            "status": "success",
            "type": "homepage",
            "pdf_path": parse_output["output_dir"],
            "template_used": template_path.name,
            "output_html": str(final_html_path),
            "output_dir": str(output_dir),
            "assets_dir": str(generator.output_assets_dir) if hasattr(generator, 'output_assets_dir') else str(output_dir / "assets"),
            "planned_sections": planned_sections,
            "parse_output": parse_output["output_dir"],
            "bge_output": bge_output["output_file"]
        }
        
        print(f"\n✅ 主页生成完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"🌐 主页文件: {result['output_html']}")
        
        return result
    
    def _run_explanation_generation_step(
        self,
        parse_output: Dict,
        bge_output: Dict,
        output_dir: Path,
        paper_name: str
    ) -> Dict:
        """运行论文讲解生成步骤"""
        # 设置输出目录
        if output_dir is None:
            output_dir = Path(parse_output["output_dir"]).parent / "explanation_output"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导入生成模块
        sys.path.append(str(Path(__file__).parent))
        
        # 导入论文讲解生成器
        from tuiwen import PaperExplanationGenerator
        
        # 构建文件路径
        content_json = parse_output["content_json"]
        modules_json = bge_output["output_file"]
        images_json = parse_output["images_json"]
        tables_json = parse_output["tables_json"]
        
        print(f"🤖 初始化论文讲解生成器...")
        
        # 创建生成器实例
        generator = PaperExplanationGenerator(
            content_json_path=content_json,
            modules_json_path=modules_json,
            images_json_path=images_json,
            tables_json_path=tables_json
        )
        
        # 生成讲解文件
        output_md_path = output_dir / f"{paper_name}_explanation.md"
        output_assets_dir = output_dir / "assets"
        
        print(f"📝 生成论文讲解: {output_md_path}")
        generated_path = generator.generate_explanation(str(output_md_path))
        
        result = {
            "status": "success",
            "type": "explanation",
            "pdf_path": parse_output["output_dir"],
            "output_md": generated_path,
            "output_dir": str(output_dir),
            "assets_dir": str(output_assets_dir),
            "assets_count": len(generator.assets_mapping) if hasattr(generator, 'assets_mapping') else 0,
            "parse_output": parse_output["output_dir"],
            "bge_output": bge_output["output_file"]
        }
        
        print(f"\n✅ 论文讲解生成完成!")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 Markdown文件: {result['output_md']}")
        print(f"🖼️ 资源文件: {result['assets_count']} 个")
        
        return result

# ==========================================================
# 简化版使用接口
# ==========================================================

def create_paper_homepage(
    pdf_path: str,
    template_name: str = "orangepaper.html",
    template_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None
) -> str:
    """
    简化版函数：创建论文主页
    
    Args:
        pdf_path: PDF文件路径
        template_name: 模板文件名（可选）
        template_dir: 模板目录（可选，默认使用内置路径）
        output_dir: 输出目录（可选）
        config: 配置字典（可选）
        
    Returns:
        str: 生成的HTML文件路径
    """
    return _run_pipeline(
        pdf_path=pdf_path,
        output_type="homepage",
        template_name=template_name,
        template_dir=template_dir,
        output_dir=output_dir,
        config=config
    )

def create_paper_explanation(
    pdf_path: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None
) -> str:
    """
    简化版函数：创建论文讲解（推文）
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录（可选）
        config: 配置字典（可选）
        
    Returns:
        str: 生成的Markdown文件路径
    """
    return _run_pipeline(
        pdf_path=pdf_path,
        output_type="explanation",
        template_name=None,  # 推文生成不需要模板
        template_dir=None,
        output_dir=output_dir,
        config=config
    )

def _run_pipeline(
    pdf_path: str,
    output_type: str,
    template_name: Optional[str] = None,
    template_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[Dict] = None
) -> str:
    """
    运行流水线的内部函数
    
    Args:
        pdf_path: PDF文件路径
        output_type: 输出类型 - "homepage" 或 "explanation"
        template_name: 模板文件名（仅主页生成需要）
        template_dir: 模板目录（可选）
        output_dir: 输出目录（可选）
        config: 配置字典（可选）
        
    Returns:
        str: 生成的输出文件路径
    """
    # 合并配置
    default_config = {
        "bge_model_path": "/home/gaojuanru/.cache/huggingface/BAAI/bge-m3",
        "qwen_model_path": "/home/gaojuanru/.cache/huggingface/Qwen/Qwen2.5-7B-Instruct",
        "csv_path": "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/sample_papers_updated.csv",
        "llm2_model": "Qwen2.5-7B-Instruct"
    }
    
    if config:
        default_config.update(config)
    
    # 使用默认模板目录
    if template_dir is None:
        template_dir = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/muban_clean"
    
    # 创建流水线
    pipeline = PaperHomepagePipeline(
        template_dir=template_dir,
        bge_model_path=default_config["bge_model_path"],
        qwen_model_path=default_config["qwen_model_path"],
        csv_path=default_config["csv_path"],
        llm2_model=default_config["llm2_model"]
    )
    
    # 运行流水线
    result = pipeline.run_full_pipeline(
        pdf_path=pdf_path,
        output_type=output_type,
        template_name=template_name if template_name else "orangepaper.html",
        output_dir=output_dir
    )
    
    if result["status"] == "success":
        if output_type == "homepage":
            return result["output_html"]
        else:  # explanation
            return result["output_md"]
    else:
        raise RuntimeError(f"生成失败: {result.get('error', '未知错误')}")

# ==========================================================
# 命令行接口
# ==========================================================

def main():
    """命令行入口点"""
    import argparse
    
    parser = argparse.ArgumentParser(description="论文主页/讲解生成流水线")
    parser.add_argument("pdf_path", help="PDF文件路径")
    parser.add_argument("--type", choices=["homepage", "explanation"], default="homepage", 
                       help="生成类型：homepage（主页）或 explanation（推文讲解）")
    parser.add_argument("--template", default="orangepaper.html", help="模板文件名（仅主页生成需要）")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--template-dir", help="模板目录（仅主页生成需要）")
    parser.add_argument("--config", help="配置文件路径（JSON格式）")
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    try:
        if args.type == "homepage":
            # 生成主页
            output_file = create_paper_homepage(
                pdf_path=args.pdf_path,
                template_name=args.template,
                template_dir=args.template_dir,
                output_dir=args.output,
                config=config
            )
            print(f"\n✅ 主页生成成功!")
            print(f"📄 输出文件: {output_file}")
        else:
            # 生成推文讲解
            output_file = create_paper_explanation(
                pdf_path=args.pdf_path,
                output_dir=args.output,
                config=config
            )
            print(f"\n✅ 论文讲解生成成功!")
            print(f"📄 输出文件: {output_file}")
        
        # 打开浏览器（可选，仅主页）
        if args.type == "homepage":
            open_in_browser = input("是否在浏览器中打开主页? (y/n): ").lower() == 'y'
            if open_in_browser:
                import webbrowser
                webbrowser.open(f"file://{output_file}")
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()