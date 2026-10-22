import io
import os
import re
import pymupdf as fitz
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from app.schemas.ingest import PageLayoutInfo, DocumentSummary


class PDFParserService:
    def __init__(self):
        # Initialize Docling converter (downloads/loads models on first instantiation)
        self.docling_converter = DocumentConverter()

    def parse_with_docling(self, file_path_or_stream: str | Path | io.BytesIO) -> str:
        """
        Parses a PDF using Docling's deep learning layout engine.
        Outputs clean, structure-aware Markdown with tables and multi-column reading flow preserved.
        """
        try:
            result = self.docling_converter.convert(file_path_or_stream)
            markdown_content = result.document.export_to_markdown()
            return markdown_content
        except Exception as e:
            # Fallback logger or print statement
            print(f"Docling parsing failed: {e}. Falling back to PyMuPDF raw extraction.")
            return ""

    @staticmethod
    def count_real_tables(markdown_text: str) -> int:
        """
        Counts distinct, valid Markdown tables while ignoring single layout lines, 
        metadata grids, copyright blocks, and sub-header divider splits.
        """
        if not markdown_text:
            return 0

        lines = markdown_text.split("\n")
        table_count = 0
        in_table = False
        current_table_rows = 0
        current_table_text = []

        # Patterns commonly found in false-positive layout grids (copyright, header blocks, author details)
        ignore_keywords = [
            "acm acknowledges", 
            "copyright", 
            "arxiv:", 
            "department of", 
            "association for computing machinery",
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
                    # Validate if this block represents a genuine data table
                    full_table_str = "\n".join(current_table_text).lower()
                    
                    # Rule 1: A valid Markdown table requires at least 3 pipe rows (| Header |, |---| , | Data |)
                    # Rule 2: Exclude layout metadata/copyright grids
                    is_metadata_grid = any(kw in full_table_str for kw in ignore_keywords)
                    
                    if current_table_rows >= 3 and not is_metadata_grid:
                        table_count += 1

                    in_table = False
                    current_table_rows = 0
                    current_table_text = []

        # Check for any remaining table at end of document
        if in_table and current_table_rows >= 3:
            full_table_str = "\n".join(current_table_text).lower()
            if not any(kw in full_table_str for kw in ignore_keywords):
                table_count += 1

        return table_count

    @staticmethod
    def extract_abstract_fallback(file_bytes: bytes, filename: str) -> Optional[Document]:
        """
        Regex fallback: Scans the first 4 pages for Abstract section and guarantees priority indexing.
        """
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
        """
        Regex fallback: Scans the first 4 pages for Keywords / Index Terms section.
        """
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
        Main pipeline orchestrator for a single PDF:
        1. Uses Docling to parse complex document structure into Markdown.
        2. Scans for priority sections (Abstract, Keywords).
        3. Constructs structured LangChain Documents and metadata summaries.
        """
        documents: List[Document] = []
        
        # 1. Scanned / Priority Extraction (Abstract & Keywords)
        abstract_doc = self.extract_abstract_fallback(file_bytes, filename)
        keywords_doc = self.extract_keywords_fallback(file_bytes, filename)
        
        if abstract_doc:
            documents.append(abstract_doc)
        if keywords_doc:
            documents.append(keywords_doc)

        # 2. Docling High-Precision Parsing
        docling_md = self.parse_with_docling(temp_filepath)
        
        if docling_md:
            documents.append(
                Document(
                    page_content=docling_md,
                    metadata={
                        "source": filename,
                        "type": "body_docling",
                        "parser": "docling_v2"
                    }
                )
            )
        else:
            # Fallback PyMuPDF text extraction if Docling fails
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

        # 3. Calculate metrics for DocumentSummary response schema
        doc_fitz = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc_fitz)
        doc_fitz.close()

        # Accurate table counting filtering out false-positive layout blocks
        tables_found = self.count_real_tables(docling_md) if docling_md else 0

        summary = DocumentSummary(
            filename=filename,
            total_pages=total_pages,
            abstract_found=abstract_doc is not None,
            tables_found=tables_found,
            layouts=[
                PageLayoutInfo(
                    page=i + 1,
                    layout="docling_adaptive" if docling_md else "fallback_single"
                )
                for i in range(total_pages)
            ]
        )

        return documents, summary


# Global singleton instance
pdf_parser_service = PDFParserService()