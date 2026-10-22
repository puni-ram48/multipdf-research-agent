import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_groq import ChatGroq

from app.core.config import settings
from app.schemas.chat import SourceCitation, ChatResponse
from app.services.vector_store import vector_store_service
from app.db.database import SessionLocal
from app.db.models import ChatSessionMessage

# Load environment variables from .env file
load_dotenv()


def save_chat_turn_to_db(session_id: str, question: str, answer: str, score: Optional[float], citations: List[SourceCitation]) -> None:
    """Persists a Q&A conversation turn, evaluation score, and citations to PostgreSQL."""
    try:
        db = SessionLocal()
        citation_dicts = [
            {
                "source": c.source,
                "page": c.page,
                "section": c.section,
                "type": c.type,
                "layout": c.layout
            }
            for c in citations
        ]

        record = ChatSessionMessage(
            session_id=session_id,
            user_question=question,
            assistant_answer=answer,
            faithfulness_score=score,
            citations=citation_dicts
        )
        db.add(record)
        db.commit()
        db.close()
    except Exception as e:
        print(f"⚠️ Failed to persist chat turn to PostgreSQL: {e}")


class RAGEngineService:
    def __init__(self):
        # In-memory session conversation history buffer
        # In a distributed setup, this can be swapped with Redis
        self.session_memory: Dict[str, List[Dict[str, str]]] = {}

    def get_memory(self, session_id: str) -> List[Dict[str, str]]:
        return self.session_memory.get(session_id, [])

    def add_to_memory(self, session_id: str, question: str, answer: str) -> None:
        if session_id not in self.session_memory:
            self.session_memory[session_id] = []

        self.session_memory[session_id].append({"question": question, "answer": answer})

        # Trim oldest turns if history exceeds max turns
        if len(self.session_memory[session_id]) > settings.MAX_MEMORY_TURNS:
            self.session_memory[session_id] = self.session_memory[session_id][-settings.MAX_MEMORY_TURNS:]

    def clear_memory(self, session_id: str) -> None:
        self.session_memory[session_id] = []

    def format_memory_for_prompt(self, session_id: str) -> str:
        memory = self.get_memory(session_id)
        if not memory:
            return ""
        lines = []
        for turn in memory:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    @staticmethod
    def compute_faithfulness_score(answer: Any, context: Any) -> float:
        """
        Computes answer grounding faithfulness score (fraction of answer keywords in context).
        """
        # Convert list/non-string answer to string safely
        if isinstance(answer, list):
            answer_text = " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in answer
            )
        else:
            answer_text = str(answer)

        if isinstance(context, list):
            context_text = " ".join(str(c) for c in context)
        else:
            context_text = str(context)

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of",
            "and", "or", "it", "this", "that", "with", "for", "not", "from", "as", "by",
            "be", "has", "have", "do", "does", "they", "we", "our", "their", "which",
            "also", "can", "used", "using", "based", "paper", "show", "shows", "shown"
        }

        words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer_text.lower())) - stopwords
        if not words:
            return 1.0
        matched = sum(1 for w in words if w in context_text.lower())
        return round(min(matched / len(words), 1.0), 2)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _invoke_chain(self, chain, question: str):
        """Helper execution wrapper with automatic exponential backoff retry."""
        return chain.invoke(question)

    def answer_question(self, session_id: str, question: str) -> ChatResponse:
        db = vector_store_service.load_index(session_id)
        if db is None:
            return ChatResponse(
                question=question,
                answer="⚠️ No document index found for this session. Please upload research papers first.",
                sources=[],
                faithfulness_score=None
            )

        groq_api_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
        if not groq_api_key:
            return ChatResponse(
                question=question,
                answer="⚠️ GROQ_API_KEY is missing. Please set your GROQ_API_KEY in the .env file.",
                sources=[],
                faithfulness_score=None
            )

        # 1. Retrieve top-K relevant document chunks
        retriever = db.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K})
        source_docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in source_docs)

        # 2. Format conversation history
        history_text = self.format_memory_for_prompt(session_id)
        history_block = (
            f"\n\nConversation history so far:\n{history_text}\n"
            if history_text else ""
        )

        # 3. Construct prompt
        prompt = PromptTemplate(
            template="""You are an expert research assistant. Use ONLY the context below to answer the question about the research paper(s).
{history_block}
Guidelines:
- If the answer involves metrics, numbers, or key results — quote them exactly.
- State the section (Abstract, Methodology, Results, Conclusion) the answer originates from.
- For comparison questions, address each paper separately before comparing.
- Resolve references like 'that', 'it', or 'this' using conversation history above.
- If the answer is not contained within the provided context, state:
"This information is not available in the provided research papers."
- Do NOT hallucinate or infer facts not supported by context.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question", "history_block"]
        )

        # 4. Invoke LLM (Groq LLM)
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,
            api_key=groq_api_key
        )

        rag_chain = (
            RunnableMap({
                "context": lambda q: context,
                "question": RunnablePassthrough(),
                "history_block": lambda q: history_block,
            })
            | prompt
            | llm
        )

        response = self._invoke_chain(rag_chain, question)

        # Ensure answer is converted to a string if response.content is a list
        if isinstance(response.content, list):
            answer = " ".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in response.content
            )
        else:
            answer = str(response.content)

        # 5. Calculate faithfulness score & save to memory
        faithfulness = self.compute_faithfulness_score(answer, context)
        self.add_to_memory(session_id, question, answer)

        # 6. Build deduplicated citations
        seen_citations = set()
        citations: List[SourceCitation] = []
        for doc in source_docs:
            src = doc.metadata.get("source", "Unknown")
            pg = doc.metadata.get("page", 1)
            sec = doc.metadata.get("section", "Body")
            doc_type = doc.metadata.get("type", "text")
            layout = doc.metadata.get("layout", "adaptive")

            key = (src, pg, sec)
            if key not in seen_citations:
                seen_citations.add(key)
                citations.append(
                    SourceCitation(
                        source=src,
                        page=pg,
                        section=sec,
                        type=doc_type,
                        layout=layout
                    )
                )

        # 7. Persist conversation turn to PostgreSQL
        save_chat_turn_to_db(
            session_id=session_id,
            question=question,
            answer=answer,
            score=faithfulness,
            citations=citations
        )

        return ChatResponse(
            question=question,
            answer=answer,
            sources=citations,
            faithfulness_score=faithfulness
        )


# Global singleton instance
rag_engine_service = RAGEngineService()