import io
import os
import re
import pymupdf as fitz
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from langchain_core.documents import Document

from app.core.config import settings
from app.schemas.ingest import PageLayoutInfo, DocumentSummary

# Assign CPU thread budget
os.environ["OMP_NUM_THREADS"] = str(settings.NUM_CPU_THREADS)


class PDFParserService:
    def __init__(self):
        # Configure speed-optimized digital PDF pipeline by default
        self.digital_pipeline_options = PdfPipelineOptions()
        self.digital_pipeline_options.do_ocr = False
        self.digital_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=settings.NUM_CPU_THREADS
        )

        # Configure OCR-enabled pipeline for scanned/handwritten pages
        self.ocr_pipeline_options = PdfPipelineOptions()
        self.ocr_pipeline_options.do_ocr = True
        self.ocr_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=settings.NUM_CPU_THREADS
        )

        # Fast digital converter (pre-loaded)
        self.digital_converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=self.digital_pipeline_options)
            }
        )

        # OCR converter (initialized on demand for scanned/handwritten files)
        self._ocr_converter: Optional[DocumentConverter] = None

    @property
    def ocr_converter(self) -> DocumentConverter:
        """Lazy load OCR converter only when a scanned document is uploaded."""
        if self._ocr_converter is None:
            self._ocr_converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=self.ocr_pipeline_options)
                }
            )
        return self._ocr_converter

    @staticmethod
    def is_scanned_or_handwritten(file_bytes: bytes) -> bool:
        """
        Fast pre-check (<5ms): Inspects if the document has a digital text layer.
        Returns True if the document is scanned/handwritten (requires OCR).
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            has_digital_text = any(len(page.get_text().strip()) > 50 for page in list(doc)[:3])
            doc.close()
            return not has_digital_text
        except Exception:
            return False

    def parse_with_docling(self, temp_filepath: str, is_scanned: bool = False) -> str:
        """
        Parses PDF using Docling's layout engine. Uses fast digital mode for research papers,
        and dynamically activates full-page OCR for scanned or handwritten documents.
        """
        try:
            converter = self.ocr_converter if is_scanned else self.digital_converter
            result = converter.convert(temp_filepath)
            return result.document.export_to_markdown()
        except Exception as e:
            print(f"Docling parsing failed: {e}. Falling back to PyMuPDF raw extraction.")
            return ""

    @staticmethod
    def count_real_tables(markdown_text: str) -> int:
        """Counts distinct, valid Markdown tables while ignoring layout metadata grids."""
        if not markdown_text:
            return 0

        lines = markdown_text.split("\n")
        table_count = 0
        in_table = False
        current_table_rows = 0
        current_table_text = []

        ignore_keywords = [
            "acm acknowledges", 
            "copyright", 
            "arxiv:", 
            "department of", 
            "permission block"
        ]

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                if not in_table:
                    in_table = True
                    current_table_rows = 1
                    current_table_text = [stripped]
                else:
                    current_table_rows += 1
                    current_table_text.append(stripped)
            else:
                if in_table:
                    full_str = "\n".join(current_table_text).lower()
                    is_metadata = any(kw in full_str for kw in ignore_keywords)
                    if current_table_rows >= 3 and not is_metadata:
                        table_count += 1
                    in_table = False
                    current_table_rows = 0
                    current_table_text = []

        if in_table and current_table_rows >= 3:
            full_str = "\n".join(current_table_text).lower()
            if not any(kw in full_str for kw in ignore_keywords):
                table_count += 1

        return table_count

    @staticmethod
    def extract_abstract_fallback(file_bytes: bytes, filename: str) -> Optional[Document]:
        """Regex fallback scanning first 4 pages for Abstract section."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        raw_text = ""
        for page in list(doc)[:4]:
            raw_text += page.get_text("text") + "\n"
        doc.close()

        pattern = re.compile(
            r'(?i)abstract[\s\-—:.]*\n?(.*?)'
            r'(?=\n\s*(?:keywords?|index terms?|1[\.\s]|introduction|i[\.\s]+introduction))',
            re.DOTALL
        )
        match = pattern.search(raw_text)
        if match:
            text = re.sub(r'\s+', ' ', match.group(1).strip())
            if len(text) > 50:
                return Document(
                    page_content=f"[ABSTRACT]\n{text}",
                    metadata={"source": filename, "page": 1, "type": "abstract", "section": "Abstract"}
                )
        return None

    @staticmethod
    def extract_keywords_fallback(file_bytes: bytes, filename: str) -> Optional[Document]:
        """Regex fallback scanning first 4 pages for Keywords / Index Terms section."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        raw_text = ""
        for page in list(doc)[:4]:
            raw_text += page.get_text("text") + "\n"
        doc.close()

        pattern = re.compile(
            r'(?i)(?:keywords?|index terms?)[\s:—\-]*(.*?)'
            r'(?=\n\s*(?:1[\.\s]|introduction|i[\.\s]))',
            re.DOTALL
        )
        match = pattern.search(raw_text)
        if match:
            text = re.sub(r'\s+', ' ', match.group(1).strip())
            if len(text) > 5:
                return Document(
                    page_content=f"[KEYWORDS]\n{text}",
                    metadata={"source": filename, "page": 1, "type": "keywords", "section": "Abstract"}
                )
        return None

    def process_pdf_file(
        self, file_bytes: bytes, filename: str, temp_filepath: str
    ) -> Tuple[List[Document], DocumentSummary]:
        """
        Orchestrates processing pipeline:
        1. Fast digital text check vs. scanned/handwritten detection.
        2. Priority Abstract/Keywords extraction.
        3. Docling layout parsing with adaptive OCR routing.
        """
        documents: List[Document] = []
        
        # 1. Scanned / Handwritten Pre-Check
        is_scanned = self.is_scanned_or_handwritten(file_bytes)

        # 2. Priority Abstract & Keywords Extraction
        abstract_doc = self.extract_abstract_fallback(file_bytes, filename)
        keywords_doc = self.extract_keywords_fallback(file_bytes, filename)
        
        if abstract_doc:
            documents.append(abstract_doc)
        if keywords_doc:
            documents.append(keywords_doc)

        # 3. Adaptive Docling Parsing (Digital = Fast Mode, Scanned = OCR Mode)
        docling_md = self.parse_with_docling(temp_filepath, is_scanned=is_scanned)
        
        if docling_md:
            documents.append(
                Document(
                    page_content=docling_md,
                    metadata={
                        "source": filename,
                        "type": "body_docling",
                        "parser": "docling_ocr" if is_scanned else "docling_digital"
                    }
                )
            )
        else:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "type": "text_fallback",
                                "layout": "single"
                            }
                        )
                    )
            doc.close()

        doc_fitz = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc_fitz)
        doc_fitz.close()

        tables_found = self.count_real_tables(docling_md) if docling_md else 0

        summary = DocumentSummary(
            filename=filename,
            total_pages=total_pages,
            abstract_found=abstract_doc is not None,
            tables_found=tables_found,
            layouts=[
                PageLayoutInfo(
                    page=i + 1,
                    layout="docling_ocr" if is_scanned else "docling_adaptive"
                )
                for i in range(total_pages)
            ]
        )

        return documents, summary


# Global singleton instance
pdf_parser_service = PDFParserService()