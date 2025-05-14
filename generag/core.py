"""Core module for Gene RAG system."""

import os
import json
import pickle
import re
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache

from .config import Config
from .utils import call_llm, parse_json_response


class GeneRAG:
    """Gene RAG system for genetic research question answering."""
    
    def __init__(
        self,
        db_dir: Optional[str] = None,
        citation_dir: Optional[str] = None,
        config: Optional[Config] = None,
        verbose: bool = False
    ):
        """Initialize Gene RAG system.
        
        Args:
            db_dir: Path to vector database directory
            citation_dir: Path to citation directory
            config: Configuration object (overrides db_dir and citation_dir)
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        
        if config:
            self.config = config
        else:
            self.config = Config()
            if db_dir:
                self.config.db_dir = db_dir
            if citation_dir:
                self.config.citation_dir = citation_dir
                
        self.gene_list = []
        self.gene_index = {}
        self.batch_docstores = {}
        self.citation_links = {}
        
        # Caching for gene queries
        self._gene_doc_cache = {}
        self._cache_enabled = getattr(self.config, 'cache_enabled', True)
        
        self._load_gene_index()
        self._load_citation_links()
        
    def _log(self, message: str):
        """Print debug message if verbose mode is on."""
        if self.verbose:
            print(f"[GeneRAG] {message}")
    
    def _load_gene_index(self):
        """Load gene index from database."""
        try:
            gene_index_path = os.path.join(self.config.db_dir, "gene_index.json")
            
            if not os.path.exists(gene_index_path):
                raise FileNotFoundError(f"Gene index not found at {gene_index_path}")
                
            with open(gene_index_path, 'r') as f:
                self.gene_index = json.load(f)
                
            self.gene_list = list(self.gene_index.keys())
            
        except Exception as e:
            raise RuntimeError(f"Failed to load gene index: {e}")
    
    def _load_citation_links(self):
        """Load citation links for genes."""
        if not self.config.citation_dir or not os.path.exists(self.config.citation_dir):
            self._log("Citation directory not found or not specified")
            return
            
        for file_name in os.listdir(self.config.citation_dir):
            if file_name.endswith(("_cite.txt", "_long_cite.txt")):
                gene_name = file_name.split("_")[0]
                file_path = os.path.join(self.config.citation_dir, file_name)
                
                citations = {}
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and "]" in line:
                                parts = line.split("] ", 1)
                                if len(parts) == 2 and parts[0].startswith("["):
                                    index = parts[0][1:]
                                    url = parts[1]
                                    citations[index] = url
                    
                    self.citation_links[gene_name] = citations
                except Exception:
                    continue
    
    def _preload_batch_docstores(self, genes: List[str]):
        """Preload multiple gene docstores to reduce IO operations."""
        batches_to_load = set()
        
        for gene in genes:
            if gene in self.gene_index:
                batch = self.gene_index[gene].get("batch")
                if batch and batch not in self.batch_docstores:
                    batches_to_load.add(batch)
        
        self._log(f"Preloading {len(batches_to_load)} batch docstores")
        
        # Batch load docstores
        for batch in batches_to_load:
            try:
                docstore_path = os.path.join(self.config.db_dir, batch, "docstore.pkl")
                with open(docstore_path, 'rb') as f:
                    self.batch_docstores[batch] = pickle.load(f)
                self._log(f"Loaded docstore for batch {batch}")
            except Exception as e:
                self._log(f"Error loading batch {batch}: {e}")
    
    def process_query(self, query: str, max_docs: int = 30) -> str:
        """Process a query and return a response.
        
        Args:
            query: The research question
            max_docs: Maximum number of documents to retrieve for context.
                      More documents = more comprehensive but slower response.
                      Recommended: 10-20 for quick answers, 30-50 for detailed research.
            
        Returns:
            Generated response based on retrieved documents
            
        Raises:
            ValueError: If query is empty or max_docs is invalid
        """
        # Input validation
        if not query or not query.strip():
            return "Error: Query cannot be empty."
        
        if max_docs < 1:
            raise ValueError("max_docs must be at least 1")
        
        if max_docs > 100:
            self._log(f"Warning: max_docs={max_docs} may result in slow response")
        
        self._log(f"Processing query: {query}")
        self._log(f"Max documents requested: {max_docs}")
        
        if not self.gene_list:
            return "Error: Gene database index is not loaded."
        
        # Translate query if needed
        query_en = self._translate_to_english(query)
        self._log(f"English query: {query_en}")
        
        # Identify genes in query
        explicit_genes, related_genes = self._identify_genes(query_en)
        self._log(f"Explicit genes: {explicit_genes}")
        self._log(f"Related genes: {related_genes}")
        
        genes_to_search = explicit_genes if explicit_genes else related_genes
        
        if not genes_to_search:
            # Provide more helpful error message with suggestions
            suggestions = self.gene_list[:10]
            return (f"Could not identify relevant genes in your query.\n\n"
                   f"Try searching for specific gene names. Examples: {', '.join(suggestions)}")
        
        self._log(f"Genes to search: {genes_to_search}")
        
        # Preload batch docstores for performance
        self._preload_batch_docstores(genes_to_search)
        
        # Extract keywords
        keywords = self._extract_keywords(query_en)
        self._log(f"Keywords: {keywords}")
        
        # Retrieve documents
        documents = self._retrieve_documents(genes_to_search, explicit_genes, keywords, max_docs)
        self._log(f"Retrieved {len(documents)} documents (requested: {max_docs})")
        
        if not documents:
            return f"No relevant documents found for genes: {', '.join(genes_to_search)}"
            
        # Generate response
        return self._generate_response(query, query_en, documents, genes_to_search)
    
    def _translate_to_english(self, text: str) -> str:
        """Translate text to English if needed."""
        # Check if text is already English
        english_pattern = re.compile(r'[a-zA-Z0-9\s.,;:!?()\'"-]{3,}')
        english_matches = english_pattern.findall(text)
        english_chars = sum(len(match) for match in english_matches)
        
        if english_chars / max(1, len(text)) > 0.7:
            return text
            
        # Use LLM for translation
        prompt = f"""Translate the following text to English. If it's already in English, return it unchanged.

Text: "{text}"

Respond with ONLY the translated text and nothing else."""

        response = call_llm(
            prompt,
            self.config,
            system_message="You are a translator that accurately translates text to English.",
            temperature=0.1,
            max_tokens=500
        )
        
        # Clean up response
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
            
        return response
    
    def _normalize_gene_name(self, gene: str) -> str:
        """Normalize gene name for consistency.
        
        Args:
            gene: Gene name to normalize
            
        Returns:
            Normalized gene name
        """
        # Standard gene names are typically uppercase
        return gene.strip().upper()
    
    def _identify_genes(self, query: str) -> Tuple[List[str], List[str]]:
        """Identify genes mentioned in the query."""
        prompt = f"""As a genomics expert, analyze the following query and extract gene names.
        
Query: "{query}"

Return your response as a JSON object:
{{
  "explicit_genes": ["explicitly mentioned genes"],
  "related_genes": ["related genes if no explicit ones"]
}}"""

        response = call_llm(
            prompt,
            self.config,
            system_message="You are an AI genomics expert.",
            temperature=0.1,
            max_tokens=500
        )
        
        gene_info = parse_json_response(response)
        
        # Normalize and filter genes
        explicit = []
        related = []
        
        for gene in gene_info.get("explicit_genes", []):
            normalized = self._normalize_gene_name(gene)
            if normalized in self.gene_list:
                explicit.append(normalized)
                
        for gene in gene_info.get("related_genes", []):
            normalized = self._normalize_gene_name(gene)
            if normalized in self.gene_list:
                related.append(normalized)
        
        return explicit, related
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        try:
            prompt = f"""Extract 2-5 important keywords from this gene research query.

Query: "{query}"

Return only a JSON array of keywords."""

            response = call_llm(
                prompt,
                self.config,
                system_message="You are an expert in extracting keywords from scientific queries.",
                temperature=0.1,
                max_tokens=200
            )
            
            keywords = parse_json_response(response)
            if isinstance(keywords, list):
                return keywords
        except Exception as e:
            self._log(f"Keyword extraction failed: {e}")
            
        # Fallback to simple extraction
        return query.split()[:5]
    
    def _retrieve_documents(
        self,
        all_genes: List[str],
        explicit_genes: List[str],
        keywords: List[str],
        max_docs: int
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for genes."""
        self._log(f"_retrieve_documents: max_docs={max_docs}")
        
        if explicit_genes:
            # Prioritize explicit genes
            explicit_quota = int(max_docs * 0.8)
            self._log(f"Explicit gene quota: {explicit_quota}")
            explicit_docs = self._retrieve_filtered_documents(explicit_genes, keywords, explicit_quota)
            self._log(f"Retrieved {len(explicit_docs)} explicit gene documents")
            
            # Get remaining for related genes
            related_genes = [g for g in all_genes if g not in explicit_genes]
            related_quota = max_docs - len(explicit_docs)
            self._log(f"Related gene quota: {related_quota}")
            related_docs = []
            
            if related_genes and related_quota > 0:
                related_docs = self._retrieve_filtered_documents(related_genes, keywords, related_quota)
                self._log(f"Retrieved {len(related_docs)} related gene documents")
                
            total_docs = explicit_docs + related_docs
            self._log(f"Total documents retrieved: {len(total_docs)}")
            return total_docs
        else:
            docs = self._retrieve_filtered_documents(all_genes, keywords, max_docs)
            self._log(f"Retrieved {len(docs)} documents (no explicit genes)")
            return docs
    
    def _get_gene_doc_cache_key(self, gene: str, keywords_str: str, max_docs: int) -> str:
        """Generate cache key for gene document retrieval."""
        return f"{gene}:{keywords_str}:{max_docs}"
    
    def _retrieve_filtered_documents(
        self,
        genes: List[str],
        keywords: List[str],
        max_docs: int
    ) -> List[Dict[str, Any]]:
        """Retrieve and filter documents for genes."""
        self._log(f"_retrieve_filtered_documents: genes={genes}, max_docs={max_docs}")
        
        all_documents = []
        docs_per_gene = max(1, max_docs // len(genes))
        self._log(f"Documents per gene: {docs_per_gene}")
        
        # Create keywords string for cache key
        keywords_str = ",".join(sorted(keywords)) if keywords else ""
        
        for gene in genes:
            # Check cache first
            cache_key = self._get_gene_doc_cache_key(gene, keywords_str, docs_per_gene)
            if self._cache_enabled and cache_key in self._gene_doc_cache:
                self._log(f"Using cached documents for gene {gene}")
                all_documents.extend(self._gene_doc_cache[cache_key])
                continue
            if gene not in self.gene_index:
                self._log(f"Gene {gene} not found in index")
                continue
                
            gene_info = self.gene_index[gene]
            batch = gene_info.get("batch")
            
            if not batch:
                self._log(f"No batch info for gene {gene}")
                continue
                
            self._log(f"Processing gene {gene} from batch {batch}")
            
            # Load docstore
            if batch not in self.batch_docstores:
                try:
                    docstore_path = os.path.join(self.config.db_dir, batch, "docstore.pkl")
                    if not os.path.exists(docstore_path):
                        self._log(f"Docstore not found at {docstore_path}")
                        continue
                        
                    with open(docstore_path, 'rb') as f:
                        self.batch_docstores[batch] = pickle.load(f)
                    self._log(f"Loaded docstore for batch {batch}")
                except Exception as e:
                    self._log(f"Error loading docstore for batch {batch}: {e}")
                    continue
            
            docstore = self.batch_docstores[batch]
            
            # Find documents for this gene
            gene_docs = []
            for doc_id, doc in docstore.items():
                if doc.get("metadata", {}).get("gene_id") == gene:
                    score = self._score_document(doc, keywords)
                    gene_docs.append((doc, score))
            
            self._log(f"Found {len(gene_docs)} documents for gene {gene}")
            
            # Sort by relevance
            gene_docs.sort(key=lambda x: x[1], reverse=True)
            selected_docs = [doc for doc, _ in gene_docs[:docs_per_gene]]
            self._log(f"Selected {len(selected_docs)} documents for gene {gene}")
            
            # Cache the results
            if self._cache_enabled:
                self._gene_doc_cache[cache_key] = selected_docs
                
            all_documents.extend(selected_docs)
        
        self._log(f"Total documents retrieved: {len(all_documents)}")
        return all_documents
    
    def _score_document(self, doc: Dict[str, Any], keywords: List[str]) -> float:
        """Score document relevance based on keywords.
        
        Args:
            doc: Document to score
            keywords: Keywords to search for
            
        Returns:
            Relevance score (higher is better)
        """
        if not keywords:
            return 1.0
            
        content = doc["page_content"].lower()
        content_length = len(content)
        score = 0.0
        
        # Weight keywords by position and frequency
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Count occurrences
            count = content.count(keyword_lower)
            score += count
            
            # Bonus for keywords in first 20% of document
            first_part = content[:int(content_length * 0.2)]
            if keyword_lower in first_part:
                score += 3
                
            # Smaller bonus for keywords in first half
            first_half = content[:int(content_length * 0.5)]
            if keyword_lower in first_half:
                score += 1
        
        # Normalize by document length to avoid bias towards longer documents
        if content_length > 0:
            score = score / (content_length / 1000)
        
        return score
    
    def _extract_citation_description(self, content: str, max_length: int = 100) -> str:
        """Extract a meaningful description from document content.
        
        Args:
            content: Document content
            max_length: Maximum length of description
            
        Returns:
            Descriptive text for citation
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', content)
        if not sentences:
            return content[:max_length]
        
        # Keywords that indicate key findings
        finding_keywords = [
            'found', 'show', 'demonstrate', 'indicate', 'suggest', 'reveal',
            'discover', 'prove', 'confirm', 'establish', 'identify'
        ]
        
        # Look for sentences with key findings
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in finding_keywords):
                cleaned = sentence.strip()
                if len(cleaned) > max_length:
                    return cleaned[:max_length-3] + "..."
                return cleaned
        
        # Fallback: use first sentence
        first_sentence = sentences[0].strip()
        if len(first_sentence) > max_length:
            return first_sentence[:max_length-3] + "..."
        return first_sentence
    
    def _generate_response(
        self,
        original_query: str,
        query_en: str,
        documents: List[Dict[str, Any]],
        genes: List[str]
    ) -> str:
        """Generate response using LLM."""
        self._log(f"_generate_response: {len(documents)} documents for genes {genes}")
        
        # Create descriptive citations
        citations = {}
        context_parts = []
        citations_with_urls = 0
        
        for i, doc in enumerate(documents, 1):
            gene_id = doc["metadata"]["gene_id"]
            ref_id = doc["metadata"].get("reference_id", "N/A")
            content = doc["page_content"]
            
            # Extract meaningful description
            description = self._extract_citation_description(content)
            
            # Find PubMed URL if available
            pubmed_url = None
            if gene_id in self.citation_links and str(ref_id) in self.citation_links[gene_id]:
                pubmed_url = self.citation_links[gene_id][str(ref_id)]
                citations_with_urls += 1
            
            # Create citation entry with description and URL
            if pubmed_url:
                citations[i] = {
                    'description': description,
                    'url': pubmed_url,
                    'full': f"[{i}] {description} - {pubmed_url}"
                }
            else:
                citations[i] = {
                    'description': description,
                    'url': None,
                    'full': f"[{i}] {description}"
                }
            
            # Add to context with clear labeling
            context_parts.append(f"[Research Finding {i}: {description}]\n{content}")
            
        self._log(f"Created {len(citations)} citations, {citations_with_urls} with PubMed URLs")
        
        context = "\n\n======\n\n".join(context_parts)
        
        # Format citations with clear numbering
        citations_text = ""
        citation_map = {}  # For tracking what citations are available
        
        for num, cite in citations.items():
            citations_text += cite['full'] + "\n"
            citation_map[num] = cite
            
        # Add a clear listing of available citation numbers
        available_numbers = list(range(1, len(citations) + 1))
        citations_summary = f"Available citation numbers: [1] through [{len(citations)}]\n\n"
        
        prompt = f"""You are a genetics expert writing about {', '.join(genes)} based on scientific literature.

Query: {original_query}

Research Findings:
{context}

{citations_summary}Available Citations:
{citations_text}

CRITICAL INSTRUCTIONS FOR CITATIONS:
1. Write a comprehensive response about the gene(s) and topic
2. Use markdown formatting (##, ###, **, -, etc.)
3. CITATION RULES:
   - You have citations numbered [1] through [{len(citations)}]
   - Use ONLY these numbers when citing
   - Do NOT use numbers like [15], [25], [60] etc. - these are NOT available
   - Cite relevant information using the correct available numbers
   - Each citation in text must match an available citation

4. References Section:
   - List ONLY the citations you used in the text
   - Use the exact same numbers as in the text
   - Include the full description and URL

EXAMPLE:
If you have 5 citations available, use [1], [2], [3], [4], [5] only.

Text: "THBS2 is overexpressed in CRC [1]. It activates Wnt signaling [2]..."

References:
[1] THBS2 overexpression in colorectal cancer - https://pubmed...
[2] THBS2 and Wnt pathway activation - https://pubmed...

Language: Respond in the same language as "{original_query}"
"""

        self._log("Calling LLM for response generation")
        response = call_llm(
            prompt,
            self.config,
            system_message="You are a genetics expert. Always use correct citation numbers and provide descriptive citations.",
            temperature=0.3,
            max_tokens=5000
        )
        
        # Post-process to ensure citation consistency (if available)
        try:
            from .citation_validator import post_process_response
            response = post_process_response(response, citations)
            self._log("Applied citation validation and correction")
        except ImportError:
            self._log("Citation validator not available, using raw response")
        
        return response
    
    def display_query(self, query: str, max_docs: int = 30) -> str:
        """Process a query and return the response.
        
        Args:
            query: The research question
            max_docs: Maximum number of documents to retrieve for context
            
        Returns:
            Generated response
        """
        response = self.process_query(query, max_docs)
        return response
    
    def get_available_genes(self, pattern: str = None) -> List[str]:
        """Get list of available genes, optionally filtered by pattern.
        
        Args:
            pattern: Optional regex pattern to filter genes
            
        Returns:
            List of gene names
        """
        if pattern:
            import re
            regex = re.compile(pattern, re.IGNORECASE)
            return [gene for gene in self.gene_list if regex.search(gene)]
        return self.gene_list
    
    def quick_summary(self, gene: str) -> str:
        """Get a quick summary of a gene without full query processing.
        
        Args:
            gene: Gene name
            
        Returns:
            Brief summary of the gene
        """
        if gene not in self.gene_list:
            return f"Gene {gene} not found in database."
            
        # Get just 3 documents
        docs = self._retrieve_filtered_documents([gene], [], 3)
        
        if not docs:
            return f"No information available for {gene}."
            
        prompt = f"""Based on these research findings, provide a 3-4 sentence summary of {gene}:

{docs[0]['page_content'][:500]}...

Summary:"""
        
        return call_llm(
            prompt, 
            self.config, 
            temperature=0.3, 
            max_tokens=5000
        )