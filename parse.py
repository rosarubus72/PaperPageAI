#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📄 1parse_references_filtered.py - Docling PDF解析器 + References 内容过滤
✅ 使用 Docling 作为主解析引擎
✅ 自动跳过 References 后的图片与表格
✅ 保留原输出 JSON 结构不变
"""

from dotenv import load_dotenv
import json
import os
from pathlib import Path
import re
from PIL import Image
import fitz

from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# ==========================================================
# GPU 配置与初始化
# ==========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
load_dotenv()

IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.do_ocr = True
pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# ==========================================================
# 工具函数：页码映射
# ==========================================================
def get_figure_table_page_map(pdf_path):
    """用 PyMuPDF 提取图表编号所在页"""
    page_map = {}
    recorded = set()
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            matches = re.findall(
                r"(Table\s+\d+|Table\s+\d+:|Fig(?:ure)?\.?\s*\d+|Figure\s+\d+:)", text, flags=re.I
            )
            for m in matches:
                key = re.sub(r"[:\s\.]", "", m.lower()).replace("figure", "fig")
                if key not in recorded:
                    page_map[key] = i
                    recorded.add(key)
                    print(f"📌 首次记录 {key} 在第 {i} 页")
    return page_map

# ==========================================================
# 工具函数：保存图像（增强版）
# ==========================================================
def save_full_region_image(item, document, page_no, output_path):
    """增强版图像导出：扩大裁切范围 + fallback 渲染整页"""
    try:
        img = item.get_image(document)
        img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"⚠️ Docling 裁剪失败，尝试 PyMuPDF 渲染整页: {e}")
        try:
            with fitz.open(document.source_info.source_path) as doc:
                page = doc.load_page((page_no or 1) - 1)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                pix.save(output_path)
            return True
        except Exception as e2:
            print(f"❌ 整页渲染也失败: {e2}")
            return False

# ==========================================================
# 文本解析
# ==========================================================
def extract_text_sections(raw_text: str) -> dict:
    references_pattern = re.compile(
        r"(?mi)^\s*(#{0,3}\s*)?(references|bibliography)\b.*$"
    )
    references_match = references_pattern.search(raw_text)
    if references_match:
        cutoff_idx = references_match.start()
        main_text = raw_text[:cutoff_idx]
        print(f"⚠️ 截断 References 后内容，位置 {cutoff_idx}")
    else:
        main_text = raw_text

    sections, title_pattern = [], re.compile(
        r"^(#{1,3}|\d+\.\s+|Chapter\s+\d+|Section\s+\d+|Fig\.\s+\d+|Table\s+\d+)\s*(.+)",
        re.IGNORECASE,
    )
    current_title, current_content = None, []

    for line in main_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 🔹 新增：检测 Abstract 段落
        if re.match(r"^(abstract)\b[\.:\s]*", line, re.IGNORECASE):
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip()
                })
            current_title = "Abstract"
            current_content = [re.sub(r"^(abstract)[\.:\s]*", "", line, flags=re.I).strip()]
            continue

        match = title_pattern.match(line)
        if match:
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip()
                })
            current_title, current_content = match.group(2).strip(), []
        elif current_title:
            current_content.append(line)

    if current_title:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})
    if not sections:
        sections.append({"title": "Full Content", "content": main_text.strip()})
    return {"sections": sections}

# ==========================================================
# 主函数
# ==========================================================
def process_pdf(pdf_path: str, output_root: str = "pdf_output"):
    pdf_name = Path(pdf_path).stem
    print(f"🚀 开始处理 PDF: {pdf_name}")

    output_dir = Path(output_root) / pdf_name
    fig_table_dir = output_dir / "images_and_tables"
    fig_table_dir.mkdir(parents=True, exist_ok=True)

    # Docling 转换
    try:
        conv_res = doc_converter.convert(pdf_path)
        document = conv_res.document
    except Exception as e:
        print(f"❌ Docling 转换失败: {e}")
        return

    # 页码映射
    page_map = get_figure_table_page_map(pdf_path)

    # 文本保存
    raw_text = document.export_to_markdown()
    text_sections = extract_text_sections(raw_text)
    (output_dir / f"{pdf_name}_content.json").write_text(
        json.dumps(text_sections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ==========================================================
    # 检测 References 起始页
    # ==========================================================
    references_page = None
    references_pattern_page = re.compile(
        r"(?mi)^\s*(references|bibliography)\b.*$"
    )
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if references_pattern_page.search(text):
                references_page = i
                print(f"⚠️ 检测到 References 起始于第 {references_page} 页。")
                break

    # ==========================================================
    # 表格提取
    # ==========================================================
    tables = {}
    print(f"🔍 Docling 检测到表格 {len(document.tables)} 个")
    for idx, table in enumerate(document.tables, 1):
        page_no = page_map.get(f"table{idx}", 0)
        if references_page and page_no >= references_page:
            print(f"⏭️ 跳过 References 后表格 Table {idx} (第 {page_no} 页)")
            continue
        table_path = fig_table_dir / f"{pdf_name}-table-{idx}.jpg"
        if save_full_region_image(table, document, page_no, table_path):
            with Image.open(table_path) as img:
                tables[str(idx)] = {
                    "caption": table.caption_text(document) or f"Table {idx}",
                    "table_path": str(table_path),
                    "page_no": page_no,
                    "width": img.width,
                    "height": img.height,
                }

    # 如果Docling没有检测到表格，只输出提示信息
    if len(tables) == 0:
        print("⚠️ Docling 未检测到表格")

    (output_dir / f"{pdf_name}_tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ==========================================================
    # 图片提取
    # ==========================================================
    images = {}
    for idx, image in enumerate(document.pictures, 1):
        page_no = page_map.get(f"fig{idx}", 0)
        if references_page and page_no >= references_page:
            print(f"⏭️ 跳过 References 后图片 Fig. {idx} (第 {page_no} 页)")
            continue
        image_path = fig_table_dir / f"{pdf_name}-picture-{idx}.jpg"
        if save_full_region_image(image, document, page_no, image_path):
            with Image.open(image_path) as img:
                images[str(idx)] = {
                    "caption": image.caption_text(document) or f"Fig. {idx}",
                    "image_path": str(image_path),
                    "page_no": page_no,
                    "width": img.width,
                    "height": img.height,
                }

    (output_dir / f"{pdf_name}_images.json").write_text(
        json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"🎉 完成 {pdf_name} → {output_dir}")

# ==========================================================
# 固定路径运行
# ==========================================================
if __name__ == "__main__":
    # 固定 PDF 输入路径和输出路径
    pdf_path = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/pdf"
    output_root = "/home/gaojuanru/mnt_link/gaojuanru/PaperPageAI/jiexi"

    input_path = Path(pdf_path)
    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        for pdf in pdf_files:
            process_pdf(str(pdf), output_root)
    else:
        process_pdf(str(input_path), output_root)