import os
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import glob

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

    def query(
        self,
        text: str,
        n_results: int = 1,
        max_chunk_chars: int = 350,
        max_total_chars: int = 600
    ) -> str:
        """Query the vector store and return a compact formatted string of results."""
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        
        formatted_results = []
        total_chars = 0
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            compact_doc = " ".join(doc.split())
            if len(compact_doc) > max_chunk_chars:
                compact_doc = compact_doc[:max_chunk_chars].rstrip() + "..."

            snippet = f"Source: {meta['filename']} (Category: {meta['category']})\nContent: {compact_doc}"
            next_total = total_chars + len(snippet)
            if next_total > max_total_chars and formatted_results:
                break

            formatted_results.append(snippet)
            total_chars = next_total
            
        return "\n\n---\n\n".join(formatted_results)

if __name__ == "__main__":
    # Test the service
    rag = RAGService()
    print(rag.query("What are the non-disclosure terms for suppliers?"))
