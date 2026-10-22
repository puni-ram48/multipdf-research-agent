import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ChunkerService:
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 400,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Headers to preserve as metadata during Markdown splitting
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )
        self.body_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    @staticmethod
    def detect_section(text: str) -> str:
        """Infer logical paper section from text if header metadata is missing."""
        lower = text.lower()
        if "abstract" in lower[:500]:
            return "Abstract"
        if any(w in lower for w in ["introduction", "background"]):
            return "Introduction"
        if any(w in lower for w in ["related work", "literature review"]):
            return "Related Work"
        if any(w in lower for w in ["method", "approach", "architecture", "model", "framework"]):
            return "Methodology"
        if any(w in lower for w in ["result", "experiment", "evaluation", "performance", "benchmark"]):
            return "Results"
        if any(w in lower for w in ["discussion", "analysis", "ablation"]):
            return "Discussion"
        if any(w in lower for w in ["conclusion", "future work"]):
            return "Conclusion"
        if any(w in lower for w in ["reference", "bibliography"]):
            return "References"
        return "Body"

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into indexable chunks:
        - Priority documents (abstract, keywords) are kept whole.
        - Docling body documents are split by Markdown headers then character size.
        """
        priority_chunks: List[Document] = []
        body_documents: List[Document] = []

        for doc in documents:
            doc_type = doc.metadata.get("type", "")
            if doc_type in ("abstract", "keywords"):
                priority_chunks.append(doc)
            else:
                body_documents.append(doc)

        final_body_chunks: List[Document] = []

        for doc in body_documents:
            # Stage 1: Attempt Markdown structure split
            try:
                md_splits = self.markdown_splitter.split_text(doc.page_content)
                # Re-attach parent metadata (source, filename, etc.)
                for md_doc in md_splits:
                    md_doc.metadata.update(doc.metadata)
                    
                    # Extract highest header found into metadata section
                    section_name = (
                        md_doc.metadata.get("Header 3")
                        or md_doc.metadata.get("Header 2")
                        or md_doc.metadata.get("Header 1")
                        or self.detect_section(md_doc.page_content)
                    )
                    md_doc.metadata["section"] = section_name

                # Stage 2: Sub-split large Markdown blocks
                sub_chunks = self.body_splitter.split_documents(md_splits)
                final_body_chunks.extend(sub_chunks)

            except Exception:
                # Direct Recursive split fallback if text is not Markdown formatted
                fallback_chunks = self.body_splitter.split_documents([doc])
                for chunk in fallback_chunks:
                    if "section" not in chunk.metadata:
                        chunk.metadata["section"] = self.detect_section(chunk.page_content)
                final_body_chunks.extend(fallback_chunks)

        # Priority chunks (abstract/keywords) are indexed first for higher relevance
        return priority_chunks + final_body_chunks


# Global singleton instance
chunker_service = ChunkerService()