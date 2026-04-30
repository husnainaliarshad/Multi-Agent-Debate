from duckduckgo_search import DDGS
from typing import Dict, Any

class SearchTool:
    """Wrapper for DuckDuckGo Search tool using the official library."""
    
    def __init__(self):
        # We initialize inside the run method to avoid connection issues
        pass
    
    def run(self, query: str) -> str:
        """Run search query and return results."""
        try:
            print(f"[SearchTool] Searching for: {query}")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                if not results:
                    return "No search results found."
                
                formatted_results = []
                for i, r in enumerate(results, 1):
                    formatted_results.append(f"Result {i}:\nTitle: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}")
                
                return "\n\n".join(formatted_results)
        except Exception as e:
            print(f"[SearchTool] Error: {str(e)}")
            return f"Error performing search: {str(e)}"

# Global instance
search_tool = SearchTool()
