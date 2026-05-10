import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

class RAGService:
    _embedding_function = None

    def __init__(self, corpus_path: str = "LegalBench-RAG/corpus", db_path: str = "backend/data/chroma_db"):
        repo_root = Path(__file__).resolve().parents[2]
        self.corpus_path = self._resolve_path(repo_root, corpus_path)
        self.db_path = self._resolve_path(repo_root, db_path)
        self.collection_name = "legalbench"
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Use a shared embedding function instance to save memory
        if RAGService._embedding_function is None:
            print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            RAGService._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.embedding_function = RAGService._embedding_function
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        self._source_cache: Dict[str, str] = {}
        
        # Index documents if collection is empty
        if self.collection.count() == 0:
            self._index_documents()

    def _index_documents(self):
        print(f"Indexing documents from {self.corpus_path}...")
        txt_files = glob.glob(str(self.corpus_path / "**" / "*.txt"), recursive=True)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, file_path in enumerate(txt_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content.strip():
                        continue
                    
                    # Basic chunking: split by paragraphs or every 1000 characters
                    # For legal documents, paragraphs are often better
                    chunks = self._chunk_text(content)
                    
                    for j, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            "source": file_path,
                            "filename": os.path.basename(file_path),
                            "category": os.path.basename(os.path.dirname(file_path))
                        })
                        ids.append(f"doc_{i}_chunk_{j}")
                        
                        # Add in batches to avoid memory issues
                        if len(documents) >= 100:
                            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                            documents, metadatas, ids = [], [], []
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        # Add remaining documents
        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        print(f"Indexed {self.collection.count()} chunks from {len(txt_files)} documents.")

    def _resolve_path(self, repo_root: Path, path_str: str) -> Path:
        """Resolve configured paths relative to the repo root, not the process cwd."""
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate
        return (repo_root / candidate).resolve()

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Simple paragraph-based chunking."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def query_structured(
        self,
        text: str,
        n_results: int = 5,
        max_chunk_chars: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query the vector store and return structured retrieval results."""
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])
        distances = distances[0] if distances else []

        structured_results: List[Dict[str, Any]] = []
        for index, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            relative_path = self._relative_file_path(meta)
            start_char, end_char = self._locate_chunk_span(meta.get("source"), doc)
            preview = " ".join(doc.split())
            if max_chunk_chars and len(preview) > max_chunk_chars:
                preview = preview[:max_chunk_chars].rstrip() + "..."

            structured_results.append({
                "rank": index,
                "document": doc,
                "preview": preview,
                "distance": distances[index - 1] if index - 1 < len(distances) else None,
                "source": meta.get("source"),
                "relative_path": relative_path,
                "filename": meta.get("filename"),
                "category": meta.get("category"),
                "start_char": start_char,
                "end_char": end_char,
            })

        return structured_results

    def query(
        self,
        text: str,
        n_results: int = 1,
        max_chunk_chars: int = 350,
        max_total_chars: int = 600
    ) -> str:
        """Query the vector store and return a compact formatted string of results."""
        results = self.query_structured(
            text=text,
            n_results=n_results,
            max_chunk_chars=max_chunk_chars,
        )

        formatted_results = []
        total_chars = 0
        for item in results:
            snippet = (
                f"Source: {item['filename']} (Category: {item['category']})\n"
                f"Content: {item['preview']}"
            )
            next_total = total_chars + len(snippet)
            if next_total > max_total_chars and formatted_results:
                break

            formatted_results.append(snippet)
            total_chars = next_total
            
        return "\n\n---\n\n".join(formatted_results)

    def _relative_file_path(self, metadata: Dict[str, Any]) -> str:
        source = metadata.get("source")
        if source:
            try:
                return Path(source).resolve().relative_to(self.corpus_path).as_posix()
            except ValueError:
                pass

        category = metadata.get("category")
        filename = metadata.get("filename")
        if category and filename:
            return f"{category}/{filename}"
        return filename or ""

    def _locate_chunk_span(self, source_path: Optional[str], chunk_text: str) -> tuple[Optional[int], Optional[int]]:
        if not source_path or not chunk_text:
            return None, None

        content = self._get_source_text(source_path)
        if content is None:
            return None, None

        start_char = content.find(chunk_text)
        if start_char == -1:
            return None, None

        return start_char, start_char + len(chunk_text)

    def _get_source_text(self, source_path: str) -> Optional[str]:
        if source_path not in self._source_cache:
            try:
                with open(source_path, "r", encoding="utf-8") as handle:
                    self._source_cache[source_path] = handle.read()
            except OSError:
                return None

        return self._source_cache[source_path]

if __name__ == "__main__":
    # Test the service
    rag = RAGService()
    print(rag.query("What are the non-disclosure terms for suppliers?"))
