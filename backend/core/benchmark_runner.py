"""Benchmark runner for lightweight LegalBench evaluation during debates."""

import random
from typing import Dict, Any, List, Optional
from services.rag_service import RAGService
from services.legalbench_benchmark import LegalBenchBenchmarkService


class BenchmarkRunner:
    """Runs lightweight benchmark queries during debates to evaluate RAG performance."""
    
    def __init__(
        self,
        rag_service: RAGService,
        benchmark_service: Optional[LegalBenchBenchmarkService] = None,
        benchmark_path: str = "LegalBench-RAG/benchmarks"
    ):
        self.rag_service = rag_service
        self.benchmark_service = benchmark_service or LegalBenchBenchmarkService(
            rag_service=rag_service,
            benchmark_path=benchmark_path
        )
        self.available_benchmarks = self.benchmark_service.list_benchmarks()
    
    def run_lightweight(
        self,
        selected_benchmarks: Optional[List[str]] = None,
        queries_per_benchmark: int = 3
    ) -> Dict[str, Any]:
        """
        Run lightweight benchmark evaluation.
        
        Args:
            selected_benchmarks: List of benchmark names to run (None = random 1-2)
            queries_per_benchmark: Number of queries to run per benchmark
        
        Returns:
            Dictionary with benchmark results
        """
        # Select benchmarks
        if selected_benchmarks:
            benchmarks = [b for b in selected_benchmarks if b in self.available_benchmarks]
        else:
            # Randomly select 1-2 benchmarks
            num_benchmarks = random.randint(1, min(2, len(self.available_benchmarks)))
            benchmarks = random.sample(self.available_benchmarks, num_benchmarks)
        
        if not benchmarks:
            return {"error": "No valid benchmarks selected"}
        
        results = {}
        for benchmark_name in benchmarks:
            benchmark_result = self._run_single_benchmark(
                benchmark_name,
                queries_per_benchmark
            )
            results[benchmark_name] = benchmark_result
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        return {
            "benchmarks": results,
            "overall_score": overall_score,
            "benchmarks_run": list(results.keys())
        }
    
    def _run_single_benchmark(
        self,
        benchmark_name: str,
        num_queries: int
    ) -> Dict[str, Any]:
        """Run a single benchmark with limited queries."""
        try:
            tests = self.benchmark_service._load_tests(benchmark_name)
            if not tests:
                return {"error": f"No tests found for {benchmark_name}"}
            
            # Limit to num_queries
            tests = tests[:num_queries]
            
            query_results = []
            for query_index, test in enumerate(tests, start=1):
                query = test["query"]
                gold_snippets = test.get("snippets", [])
                
                # Query RAG service
                retrieved = self.rag_service.query_structured(query, n_results=5)
                
                # Calculate metrics
                gold_files = sorted({
                    self.benchmark_service._normalize_path(snippet.get("file_path", ""))
                    for snippet in gold_snippets
                    if snippet.get("file_path")
                })
                retrieved_files = [item["relative_path"] for item in retrieved if item.get("relative_path")]
                matched_files = sorted({
                    item["relative_path"] for item in retrieved
                    if item.get("relative_path") in gold_files
                })
                
                # Simple precision/recall
                precision = len(matched_files) / len(retrieved_files) if retrieved_files else 0
                recall = len(matched_files) / len(gold_files) if gold_files else 0
                
                query_results.append({
                    "query_index": query_index,
                    "precision": precision,
                    "recall": recall,
                    "retrieved_count": len(retrieved_files),
                    "matched_count": len(matched_files)
                })
            
            # Aggregate results
            avg_precision = sum(q["precision"] for q in query_results) / len(query_results)
            avg_recall = sum(q["recall"] for q in query_results) / len(query_results)
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            return {
                "num_queries": len(query_results),
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "f1_score": f1,
                "query_results": query_results
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall benchmark score from individual results."""
        f1_scores = []
        for benchmark_name, result in results.items():
            if "f1_score" in result:
                f1_scores.append(result["f1_score"])
        
        if not f1_scores:
            return 0.0
        
        return sum(f1_scores) / len(f1_scores)
