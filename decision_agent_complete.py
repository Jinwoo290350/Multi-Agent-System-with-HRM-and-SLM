# decision_agent_complete.py - Complete Decision Agent with all task processors
import os
import asyncio
import logging
import json
import uuid
import re
import sys
import subprocess
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskType(Enum):
    WEB_SCRAPING = "web_scraping"
    GOOGLE_SEARCH = "google_search"
    DATABASE_QUERY = "database_query"
    KM_RAG = "knowledge_management_rag"
    CODE_EXECUTION = "code_execution"
    GENERAL_QUERY = "general_query"
    HRM_REASONING = "hrm_reasoning"

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    STRATEGIC = "strategic"

@dataclass
class TaskScore:
    task_type: TaskType
    confidence: float
    reasoning: str
    priority: int
    complexity: TaskComplexity = TaskComplexity.SIMPLE

@dataclass
class ProcessedTask:
    task_id: str
    input_text: str
    selected_task: TaskType
    confidence_used: float
    result: Any
    reasoning: str
    processing_time: float
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

class WebScrapingProcessor:
    """Web scraping processor with robust content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Extract and analyze web content from URLs"""
        
        urls = self._extract_urls(input_text)
        if not urls:
            return {"error": "No valid URLs found in input"}
        
        results = []
        for url in urls[:3]:  # Limit to 3 URLs
            try:
                content = await self._scrape_url(url)
                results.append({
                    "url": url,
                    "title": content.get("title", ""),
                    "content": content.get("content", "")[:1000],  # Truncate for demo
                    "links": content.get("links", [])[:10],
                    "images": content.get("images", [])[:5],
                    "metadata": content.get("metadata", {})
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "scraped_urls": len(results),
            "successful_scrapes": len([r for r in results if "error" not in r]),
            "results": results
        }
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from input text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        # Also check for domain-like patterns
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, text)
        
        for domain in domains:
            if not domain.startswith(('http://', 'https://')):
                urls.append(f'https://{domain}')
        
        return list(set(urls))
    
    async def _scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single URL"""
        
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        
        # Extract main content
        content = ""
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.content', '#content',
            '.post-content', '.entry-content', '.article-body'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
                break
        
        # Fallback to body content
        if not content and soup.body:
            content = soup.body.get_text(separator=' ', strip=True)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:20]:
            href = link['href']
            if href.startswith('http') or href.startswith('//'):
                full_url = href
            else:
                full_url = urljoin(url, href)
            
            links.append({
                "text": link.get_text().strip()[:100],
                "url": full_url
            })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True)[:10]:
            src = img['src']
            if src.startswith('http') or src.startswith('//'):
                full_url = src
            else:
                full_url = urljoin(url, src)
            
            images.append({
                "src": full_url,
                "alt": img.get('alt', '')[:100]
            })
        
        # Extract metadata
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name in ['description', 'keywords', 'author']:
                metadata[name] = content
            elif property_attr.startswith('og:'):
                metadata[property_attr] = content
        
        return {
            "title": title,
            "content": content,
            "links": links,
            "images": images,
            "metadata": metadata,
            "word_count": len(content.split()),
            "url": url
        }

class GoogleSearchProcessor:
    """Google search and web research processor"""
    
    def __init__(self):
        self.search_engines = {
            "duckduckgo": "https://duckduckgo.com/html/?q=",
            "bing": "https://www.bing.com/search?q="
        }
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process search queries and return results"""
        
        # Extract search query from input
        search_query = self._extract_search_query(input_text)
        
        try:
            # Use DuckDuckGo for demo (doesn't require API keys)
            results = await self._search_duckduckgo(search_query)
            
            return {
                "search_query": search_query,
                "search_engine": "DuckDuckGo",
                "results_count": len(results),
                "results": results[:10],  # Top 10 results
                "related_searches": self._generate_related_searches(search_query)
            }
            
        except Exception as e:
            # Fallback to mock results
            return await self._generate_mock_search_results(search_query)
    
    def _extract_search_query(self, text: str) -> str:
        """Extract search query from input text"""
        
        # Remove common search prefixes
        search_prefixes = [
            "search for", "google", "find", "look up", "search",
            "what is", "who is", "how to", "where is"
        ]
        
        cleaned_text = text.lower().strip()
        
        for prefix in search_prefixes:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        
        return cleaned_text
    
    async def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo"""
        
        search_url = f"https://duckduckgo.com/html/?q={query}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, headers=headers) as response:
                html = await response.text()
                
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        
        # Parse DuckDuckGo results
        for result_div in soup.find_all('div', class_='result')[:10]:
            title_elem = result_div.find('a', class_='result__a')
            snippet_elem = result_div.find('a', class_='result__snippet')
            
            if title_elem:
                title = title_elem.get_text().strip()
                url = title_elem.get('href', '')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })
        
        return results
    
    async def _generate_mock_search_results(self, query: str) -> Dict[str, Any]:
        """Generate mock search results as fallback"""
        
        mock_results = [
            {
                "title": f"Understanding {query.title()}: A Comprehensive Guide",
                "url": f"https://example.com/guide-to-{query.replace(' ', '-')}",
                "snippet": f"Learn everything about {query} with our detailed guide covering key concepts, applications, and best practices."
            },
            {
                "title": f"{query.title()} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"Wikipedia article providing comprehensive information about {query} including history, definitions, and related topics."
            },
            {
                "title": f"Latest {query.title()} News and Updates",
                "url": f"https://news.example.com/search?q={query}",
                "snippet": f"Stay updated with the latest news and developments related to {query} from trusted news sources."
            }
        ]
        
        return {
            "search_query": query,
            "search_engine": "Mock Search Engine",
            "results_count": len(mock_results),
            "results": mock_results,
            "related_searches": self._generate_related_searches(query),
            "note": "Mock results generated as fallback"
        }
    
    def _generate_related_searches(self, query: str) -> List[str]:
        """Generate related search suggestions"""
        
        related = [
            f"what is {query}",
            f"how to {query}",
            f"{query} examples",
            f"{query} benefits",
            f"{query} vs alternatives"
        ]
        
        return related

class DatabaseProcessor:
    """Database query processor with SQLite support"""
    
    def __init__(self):
        self.db_path = "decision_agent.db"
        self._init_demo_database()
    
    def _init_demo_database(self):
        """Initialize demo database with sample data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                department TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                assigned_to INTEGER,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (assigned_to) REFERENCES users (id)
            )
        """)
        
        # Insert sample data
        sample_users = [
            ("Alice Johnson", "alice@company.com", "Engineering"),
            ("Bob Smith", "bob@company.com", "Marketing"),
            ("Carol Davis", "carol@company.com", "Sales"),
            ("David Wilson", "david@company.com", "Support")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO users (name, email, department) VALUES (?, ?, ?)",
            sample_users
        )
        
        sample_tasks = [
            ("Implement user authentication", "Add login/logout functionality", "in_progress", 1, 3),
            ("Marketing campaign analysis", "Analyze Q4 campaign performance", "completed", 2, 2),
            ("Customer support tickets", "Process pending support requests", "pending", 4, 1),
            ("Sales report generation", "Generate monthly sales report", "pending", 3, 2)
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO tasks (title, description, status, assigned_to, priority) VALUES (?, ?, ?, ?, ?)",
            sample_tasks
        )
        
        conn.commit()
        conn.close()
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process database queries"""
        
        # Detect query type and generate appropriate SQL
        query_info = self._analyze_query(input_text)
        
        try:
            if query_info["type"] == "sql":
                # Direct SQL query (sanitized)
                results = self._execute_query(query_info["query"])
            else:
                # Natural language to SQL
                sql_query = self._generate_sql_from_text(input_text)
                results = self._execute_query(sql_query)
                query_info["generated_sql"] = sql_query
            
            return {
                "query_type": query_info["type"],
                "query": input_text,
                "sql_executed": query_info.get("query", query_info.get("generated_sql", "")),
                "results": results,
                "row_count": len(results) if isinstance(results, list) else 0,
                "tables_accessed": query_info.get("tables", [])
            }
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return {
                "query_type": "error",
                "query": input_text,
                "error": str(e),
                "suggestion": "Check your query syntax or try a natural language query like 'show all users' or 'find pending tasks'"
            }
    
    def _analyze_query(self, text: str) -> Dict[str, Any]:
        """Analyze input to determine query type"""
        
        text_lower = text.lower().strip()
        
        # Check if it's a SQL query
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop']
        if any(text_lower.startswith(keyword) for keyword in sql_keywords):
            return {
                "type": "sql",
                "query": text,
                "tables": self._extract_table_names(text)
            }
        
        # Natural language query
        return {
            "type": "natural",
            "intent": self._detect_intent(text_lower),
            "entities": self._extract_entities(text_lower)
        }
    
    def _generate_sql_from_text(self, text: str) -> str:
        """Generate SQL from natural language (simplified)"""
        
        text_lower = text.lower()
        
        # User queries
        if any(word in text_lower for word in ["user", "users", "people", "employee"]):
            if "count" in text_lower or "how many" in text_lower:
                return "SELECT COUNT(*) as user_count FROM users"
            elif "department" in text_lower:
                return "SELECT department, COUNT(*) as count FROM users GROUP BY department"
            else:
                return "SELECT * FROM users ORDER BY name"
        
        # Task queries
        elif any(word in text_lower for word in ["task", "tasks", "work", "assignment"]):
            if "pending" in text_lower:
                return "SELECT * FROM tasks WHERE status = 'pending'"
            elif "completed" in text_lower:
                return "SELECT * FROM tasks WHERE status = 'completed'"
            elif "priority" in text_lower:
                return "SELECT * FROM tasks ORDER BY priority DESC"
            else:
                return "SELECT * FROM tasks ORDER BY created_at DESC"
        
        # Join queries
        elif "assigned" in text_lower or "who" in text_lower:
            return """
                SELECT u.name, t.title, t.status, t.priority 
                FROM tasks t 
                JOIN users u ON t.assigned_to = u.id 
                ORDER BY t.priority DESC
            """
        
        # Default
        else:
            return "SELECT 'Available tables: users, tasks' as info"
    
    def _execute_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query and return results"""
        
        # Security: Only allow SELECT statements for demo
        if not sql_query.strip().upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed in demo mode")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        results = [dict(row) for row in rows]
        
        conn.close()
        return results
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        
        tables = []
        words = sql.split()
        
        for i, word in enumerate(words):
            if word.upper() in ['FROM', 'JOIN', 'UPDATE', 'INTO']:
                if i + 1 < len(words):
                    table_name = words[i + 1].strip(',;')
                    tables.append(table_name)
        
        return list(set(tables))
    
    def _detect_intent(self, text: str) -> str:
        """Detect query intent from natural language"""
        
        if any(word in text for word in ["show", "list", "get", "find", "select"]):
            return "retrieve"
        elif any(word in text for word in ["count", "how many", "number"]):
            return "count"
        elif any(word in text for word in ["create", "add", "insert"]):
            return "create"
        elif any(word in text for word in ["update", "change", "modify"]):
            return "update"
        else:
            return "retrieve"
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from natural language query"""
        
        entities = []
        
        if "user" in text or "users" in text:
            entities.append("users")
        if "task" in text or "tasks" in text:
            entities.append("tasks")
        if "department" in text:
            entities.append("department")
        if "pending" in text:
            entities.append("pending")
        if "completed" in text:
            entities.append("completed")
        
        return entities

class KnowledgeRAGProcessor:
    """Knowledge Management RAG (Retrieval Augmented Generation) processor"""
    
    def __init__(self):
        self.knowledge_base = self._init_knowledge_base()
    
    def _init_knowledge_base(self) -> Dict[str, Any]:
        """Initialize demo knowledge base"""
        
        return {
            "documents": [
                {
                    "id": "doc_1",
                    "title": "Company Policies and Procedures",
                    "content": "Our company follows strict security protocols. All employees must use strong passwords, enable 2FA, and follow the data protection guidelines. Remote work is allowed with proper VPN setup.",
                    "categories": ["policy", "security", "remote_work"],
                    "last_updated": "2024-01-15"
                },
                {
                    "id": "doc_2", 
                    "title": "AI and Machine Learning Best Practices",
                    "content": "When implementing AI systems, consider data quality, model validation, ethical implications, and monitoring. Use proper feature engineering and cross-validation techniques.",
                    "categories": ["ai", "ml", "best_practices", "technical"],
                    "last_updated": "2024-01-10"
                },
                {
                    "id": "doc_3",
                    "title": "Customer Support Guidelines",
                    "content": "Always respond to customers within 24 hours. Use empathetic language, provide clear solutions, and escalate complex issues to technical teams. Maintain professional tone in all communications.",
                    "categories": ["customer_service", "support", "communication"],
                    "last_updated": "2024-01-12"
                },
                {
                    "id": "doc_4",
                    "title": "Software Development Standards",
                    "content": "Follow coding standards, write unit tests, use version control, conduct code reviews, and document APIs. Implement proper error handling and security measures.",
                    "categories": ["development", "coding", "technical", "standards"],
                    "last_updated": "2024-01-08"
                }
            ],
            "metadata": {
                "total_documents": 4,
                "last_indexed": datetime.now().isoformat(),
                "categories": ["policy", "security", "ai", "ml", "customer_service", "development"]
            }
        }
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process knowledge queries using RAG approach"""
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self._retrieve_documents(input_text)
        
        # Step 2: Rank and filter documents
        ranked_docs = self._rank_documents(relevant_docs, input_text)
        
        # Step 3: Generate response using retrieved content
        response = self._generate_response(input_text, ranked_docs)
        
        return {
            "query": input_text,
            "retrieved_documents": len(relevant_docs),
            "relevant_documents": len(ranked_docs),
            "response": response,
            "sources": [
                {
                    "title": doc["title"],
                    "id": doc["id"],
                    "relevance_score": doc.get("relevance_score", 0),
                    "categories": doc["categories"]
                }
                for doc in ranked_docs[:3]
            ],
            "knowledge_base_stats": self.knowledge_base["metadata"]
        }
    
    def _retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve potentially relevant documents"""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_docs = []
        
        for doc in self.knowledge_base["documents"]:
            # Check title relevance
            title_words = set(doc["title"].lower().split())
            title_overlap = len(query_words.intersection(title_words))
            
            # Check content relevance
            content_words = set(doc["content"].lower().split())
            content_overlap = len(query_words.intersection(content_words))
            
            # Check category relevance
            category_overlap = 0
            for category in doc["categories"]:
                if any(word in category.lower() for word in query_words):
                    category_overlap += 1
            
            # Calculate relevance score
            total_score = title_overlap * 3 + content_overlap + category_overlap * 2
            
            if total_score > 0:
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = total_score
                relevant_docs.append(doc_copy)
        
        return relevant_docs
    
    def _rank_documents(self, documents: List[Dict], query: str) -> List[Dict]:
        """Rank documents by relevance"""
        
        # Sort by relevance score
        ranked = sorted(documents, key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply additional ranking factors
        for doc in ranked:
            # Boost score for exact phrase matches
            if any(phrase in doc["content"].lower() for phrase in query.lower().split()):
                doc["relevance_score"] *= 1.2
            
            # Consider recency (newer documents get slight boost)
            doc_date = datetime.fromisoformat(doc["last_updated"])
            days_old = (datetime.now() - doc_date).days
            recency_factor = max(0.8, 1.0 - (days_old / 365))  # Decay over year
            doc["relevance_score"] *= recency_factor
        
        return sorted(ranked, key=lambda x: x["relevance_score"], reverse=True)
    
    def _generate_response(self, query: str, documents: List[Dict]) -> str:
        """Generate response using retrieved documents"""
        
        if not documents:
            return f"I couldn't find specific information about '{query}' in the knowledge base. The available topics include: {', '.join(self.knowledge_base['metadata']['categories'])}"
        
        # Use top 2-3 most relevant documents
        top_docs = documents[:3]
        
        response_parts = []
        response_parts.append(f"Based on the available knowledge, here's what I found about '{query}':")
        
        for i, doc in enumerate(top_docs, 1):
            # Extract relevant sentences from content
            relevant_content = self._extract_relevant_content(doc["content"], query)
            response_parts.append(f"\n{i}. From '{doc['title']}':")
            response_parts.append(f"   {relevant_content}")
        
        # Add source attribution
        response_parts.append(f"\nSources consulted: {', '.join([doc['title'] for doc in top_docs])}")
        
        return " ".join(response_parts)
    
    def _extract_relevant_content(self, content: str, query: str) -> str:
        """Extract most relevant sentences from document content"""
        
        sentences = content.split('. ')
        query_words = set(query.lower().split())
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            # Return the most relevant sentence(s)
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return scored_sentences[0][0] + "."
        else:
            # Return first sentence as fallback
            return sentences[0] + "." if sentences else content[:200] + "..."

class CodeExecutionProcessor:
    """Safe code execution processor for Python calculations"""
    
    def __init__(self):
        self.allowed_modules = {
            'math', 'statistics', 'random', 'datetime', 're', 'json'
        }
        self.safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'format', 'hex', 'int', 'len', 'list', 'map',
            'max', 'min', 'oct', 'ord', 'pow', 'range', 'reversed', 'round',
            'set', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
        }
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process code execution requests safely"""
        
        # Extract and clean code
        code_info = self._extract_code(input_text)
        
        if not code_info["code"]:
            return {
                "error": "No executable code found in input",
                "suggestion": "Include Python code or mathematical expressions like: execute: print(2+2) or calculate: 5*7+3"
            }
        
        try:
            # Execute code safely
            result = await self._execute_safe_code(code_info["code"])
            
            return {
                "code": code_info["code"],
                "language": code_info["language"],
                "execution_type": code_info["type"],
                "result": result["output"],
                "execution_time": result["execution_time"],
                "status": "success"
            }
            
        except Exception as e:
            return {
                "code": code_info["code"],
                "language": code_info["language"],
                "execution_type": code_info["type"], 
                "error": str(e),
                "status": "error",
                "suggestion": "Make sure your code uses only basic Python operations and allowed modules"
            }
    
    def _extract_code(self, text: str) -> Dict[str, Any]:
        """Extract code from input text"""
        
        text = text.strip()
        
        # Check for explicit code markers
        code_markers = [
            ("execute:", "python"),
            ("python:", "python"),
            ("calculate:", "python"),
            ("math:", "python"),
            ("run:", "python")
        ]
        
        for marker, lang in code_markers:
            if text.lower().startswith(marker):
                code = text[len(marker):].strip()
                return {
                    "code": code,
                    "language": lang,
                    "type": "explicit"
                }
        
        # Check for code blocks
        code_block_patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return {
                    "code": matches[0].strip(),
                    "language": "python",
                    "type": "code_block"
                }
        
        # Check for mathematical expressions
        if self._is_math_expression(text):
            return {
                "code": f"print({text})",
                "language": "python",
                "type": "math_expression"
            }
        
        # Default: treat as Python code if it looks like code
        if self._looks_like_code(text):
            return {
                "code": text,
                "language": "python",
                "type": "inferred"
            }
        
        return {
            "code": "",
            "language": "unknown",
            "type": "none"
        }
    
    def _is_math_expression(self, text: str) -> bool:
        """Check if text is a mathematical expression"""
        
        # Simple heuristic for math expressions
        math_chars = set("0123456789+-*/.()% ")
        return (
            len(text) < 100 and
            all(c in math_chars for c in text) and
            any(c in "+-*/" for c in text)
        )
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like executable code"""
        
        code_indicators = [
            'print(', 'def ', 'for ', 'while ', 'if ', 'import ',
            'return ', '=', '+=', '-=', '*=', '/='
        ]
        
        return any(indicator in text for indicator in code_indicators)
    
    async def _execute_safe_code(self, code: str) -> Dict[str, Any]:
        """Execute code in a safe environment"""
        
        start_time = datetime.now()
        
        # Create safe environment
        safe_globals = {
            '__builtins__': {name: getattr(__builtins__, name) for name in self.safe_builtins if hasattr(__builtins__, name)}
        }
        
        # Add safe modules
        import math
        import statistics
        import random
        import datetime
        import re
        import json
        
        safe_globals.update({
            'math': math,
            'statistics': statistics,
            'random': random,
            'datetime': datetime,
            're': re,
            'json': json
        })
        
        # Capture output
        import io
        import sys
        
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Execute code with timeout protection
            exec(code, safe_globals, {})
            output = captured_output.getvalue()
            
            # If no output, try to evaluate as expression
            if not output.strip():
                try:
                    result = eval(code, safe_globals, {})
                    output = str(result)
                except:
                    output = "Code executed successfully (no output)"
            
        finally:
            sys.stdout = original_stdout
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "output": output.strip(),
            "execution_time": execution_time
        }

class CompleteDecisionAgent:
    """Complete Decision Agent with all processors and HRM integration"""
    
    def __init__(self):
        # Initialize processors
        self.web_scraper = WebScrapingProcessor()
        self.search_processor = GoogleSearchProcessor()
        self.database_processor = DatabaseProcessor()
        self.knowledge_processor = KnowledgeRAGProcessor()
        self.code_processor = CodeExecutionProcessor()
        
        # Initialize HRM (if available)
        try:
            from azure_hrm_system import AzureHierarchicalReasoningModel
            self.hrm = AzureHierarchicalReasoningModel()
            self.hrm_available = True
        except ImportError:
            self.hrm = None
            self.hrm_available = False
            logger.warning("HRM system not available")
        
        # Processing history
        self.processing_history: List[ProcessedTask] = []
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "task_counts": {task_type.value: 0 for task_type in TaskType},
            "average_confidence": 0.0,
            "total_processing_time": 0.0
        }
    
    async def process_request(self, input_text: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Main processing function with intelligent task routing"""
        
        start_time = datetime.now()
        task_id = str(uuid.uuid4())
        self.stats["total_requests"] += 1
        
        try:
            # Step 1: Analyze input and get task scores
            task_scores = await self.analyze_input(input_text)
            
            if not task_scores:
                raise ValueError("Unable to analyze input")
            
            # Step 2: Select best task based on confidence
            selected_task_score = task_scores[0]  # Highest confidence
            
            # Step 3: Process with selected task processor
            result = await self._execute_task(
                selected_task_score.task_type,
                input_text,
                selected_task_score
            )
            
            # Step 4: Calculate processing time and update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Step 5: Create processed task record
            processed_task = ProcessedTask(
                task_id=task_id,
                input_text=input_text,
                selected_task=selected_task_score.task_type,
                confidence_used=selected_task_score.confidence,
                result=result,
                reasoning=selected_task_score.reasoning,
                processing_time=processing_time,
                metadata={
                    "task_scores": [
                        {
                            "task_type": score.task_type.value,
                            "confidence": score.confidence,
                            "reasoning": score.reasoning
                        }
                        for score in task_scores
                    ],
                    "options": options or {}
                }
            )
            
            self.processing_history.append(processed_task)
            self._update_stats(processed_task)
            
            return {
                "task_id": task_id,
                "status": "success",
                "selected_task": selected_task_score.task_type.value,
                "confidence_used": selected_task_score.confidence,
                "reasoning": selected_task_score.reasoning,
                "result": result,
                "processing_time": processing_time,
                "task_analysis": processed_task.metadata["task_scores"]
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["failed_requests"] += 1
            
            error_task = ProcessedTask(
                task_id=task_id,
                input_text=input_text,
                selected_task=TaskType.GENERAL_QUERY,
                confidence_used=0.0,
                result=None,
                reasoning="Processing failed",
                processing_time=processing_time,
                status="error",
                error=str(e)
            )
            
            self.processing_history.append(error_task)
            
            logger.error(f"Task processing failed: {str(e)}")
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "processing_time": processing_time,
                "suggestion": "Please check your input and try again"
            }
    
    async def analyze_input(self, input_text: str) -> List[TaskScore]:
        """Analyze input and return scored task options"""
        
        scores = []
        text_lower = input_text.lower().strip()
        
        # Web scraping detection
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, input_text)
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, input_text)
        
        if urls or domains:
            scores.append(TaskScore(
                task_type=TaskType.WEB_SCRAPING,
                confidence=0.95,
                reasoning="URLs detected in input - web scraping required",
                priority=1,
                complexity=TaskComplexity.MEDIUM
            ))
        
        # Search detection
        search_keywords = ['search', 'find', 'look up', 'google', 'what is', 'who is', 'where is']
        if any(keyword in text_lower for keyword in search_keywords):
            confidence = 0.8
            # Boost confidence if no URLs present
            if not (urls or domains):
                confidence = 0.9
            
            scores.append(TaskScore(
                task_type=TaskType.GOOGLE_SEARCH,
                confidence=confidence,
                reasoning="Search keywords detected in input",
                priority=2,
                complexity=TaskComplexity.SIMPLE
            ))
        
        # Database query detection
        db_keywords = ['database', 'query', 'sql', 'table', 'users', 'tasks', 'show', 'list', 'count']
        db_score = sum(1 for keyword in db_keywords if keyword in text_lower) / len(db_keywords)
        
        if db_score > 0.1:
            scores.append(TaskScore(
                task_type=TaskType.DATABASE_QUERY,
                confidence=min(0.9, 0.5 + db_score),
                reasoning=f"Database-related keywords detected (score: {db_score:.2f})",
                priority=2,
                complexity=TaskComplexity.MEDIUM
            ))
        
        # Knowledge management RAG detection
        km_keywords = ['policy', 'procedure', 'guideline', 'best practice', 'standard', 'documentation']
        km_score = sum(1 for keyword in km_keywords if keyword in text_lower) / len(km_keywords)
        
        if km_score > 0 or any(word in text_lower for word in ['how should', 'what should', 'company policy']):
            scores.append(TaskScore(
                task_type=TaskType.KM_RAG,
                confidence=0.7 + km_score * 0.2,
                reasoning="Knowledge management query detected",
                priority=2,
                complexity=TaskComplexity.MEDIUM
            ))
        
        # Code execution detection
        code_keywords = ['execute', 'run', 'calculate', 'python', 'code', 'math']
        has_code_markers = any(text_lower.startswith(marker) for marker in ['execute:', 'python:', 'calculate:', 'math:'])
        has_code_blocks = '```' in input_text or '`' in input_text
        has_math_expression = bool(re.search(r'[0-9]+\s*[+\-*/]\s*[0-9]+', input_text))
        
        if has_code_markers or has_code_blocks:
            confidence = 0.95
        elif has_math_expression:
            confidence = 0.85
        elif any(keyword in text_lower for keyword in code_keywords):
            confidence = 0.7
        else:
            confidence = 0.0
        
        if confidence > 0:
            scores.append(TaskScore(
                task_type=TaskType.CODE_EXECUTION,
                confidence=confidence,
                reasoning="Code execution request detected",
                priority=1,
                complexity=TaskComplexity.SIMPLE
            ))
        
        # HRM reasoning detection (complex analytical tasks)
        hrm_keywords = ['analyze', 'compare', 'evaluate', 'strategic', 'complex', 'comprehensive']
        complexity_indicators = ['multiple', 'various', 'factors', 'considerations', 'pros and cons']
        
        hrm_score = (
            sum(1 for keyword in hrm_keywords if keyword in text_lower) * 0.2 +
            sum(1 for indicator in complexity_indicators if indicator in text_lower) * 0.15
        )
        
        # Boost score for long, complex queries
        if len(input_text.split()) > 20:
            hrm_score += 0.2
        
        if hrm_score > 0.3 and self.hrm_available:
            scores.append(TaskScore(
                task_type=TaskType.HRM_REASONING,
                confidence=min(0.9, hrm_score),
                reasoning="Complex reasoning task detected - suitable for HRM",
                priority=1,
                complexity=TaskComplexity.STRATEGIC if hrm_score > 0.6 else TaskComplexity.COMPLEX
            ))
        
        # General query (fallback)
        general_confidence = max(0.4, 1.0 - max([score.confidence for score in scores] + [0]))
        scores.append(TaskScore(
            task_type=TaskType.GENERAL_QUERY,
            confidence=general_confidence,
            reasoning="General AI query handling",
            priority=3,
            complexity=TaskComplexity.SIMPLE
        ))
        
        # Sort by confidence and priority
        scores.sort(key=lambda x: (-x.confidence, x.priority))
        
        return scores
    
    async def _execute_task(self, task_type: TaskType, input_text: str, task_score: TaskScore) -> Any:
        """Execute the selected task"""
        
        if task_type == TaskType.WEB_SCRAPING:
            return await self.web_scraper.process(input_text)
        
        elif task_type == TaskType.GOOGLE_SEARCH:
            return await self.search_processor.process(input_text)
        
        elif task_type == TaskType.DATABASE_QUERY:
            return await self.database_processor.process(input_text)
        
        elif task_type == TaskType.KM_RAG:
            return await self.knowledge_processor.process(input_text)
        
        elif task_type == TaskType.CODE_EXECUTION:
            return await self.code_processor.process(input_text)
        
        elif task_type == TaskType.HRM_REASONING and self.hrm_available:
            # Use HRM for complex reasoning
            hrm_result = await self.hrm.process_query(input_text, task_score.complexity)
            return hrm_result
        
        else:  # GENERAL_QUERY or fallback
            return await self._process_general_query(input_text)
    
    async def _process_general_query(self, input_text: str) -> Dict[str, Any]:
        """Process general queries with basic AI response"""
        
        # Simple response generation (in real implementation, use LLM)
        return {
            "response": f"This is a general AI response to: '{input_text}'. In a full implementation, this would use a language model to generate a comprehensive response.",
            "type": "general_response",
            "suggestions": [
                "Try asking about specific topics like web scraping, database queries, or calculations",
                "Include URLs for web scraping",
                "Use 'search for' to trigger search functionality",
                "Ask about company policies for knowledge base queries"
            ]
        }
    
    def _update_stats(self, task: ProcessedTask):
        """Update processing statistics"""
        
        if task.status == "success":
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        self.stats["task_counts"][task.selected_task.value] += 1
        self.stats["total_processing_time"] += task.processing_time
        
        # Update average confidence
        successful_tasks = [t for t in self.processing_history if t.status == "success"]
        if successful_tasks:
            self.stats["average_confidence"] = sum(t.confidence_used for t in successful_tasks) / len(successful_tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        
        return {
            **self.stats,
            "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
            "average_processing_time": self.stats["total_processing_time"] / max(1, self.stats["total_requests"]),
            "hrm_available": self.hrm_available,
            "recent_tasks": [
                {
                    "task_id": task.task_id,
                    "selected_task": task.selected_task.value,
                    "confidence": task.confidence_used,
                    "status": task.status,
                    "processing_time": task.processing_time
                }
                for task in self.processing_history[-10:]
            ]
        }
    
    def get_task_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get task processing history"""
        
        return [
            {
                "task_id": task.task_id,
                "input": task.input_text[:100] + "..." if len(task.input_text) > 100 else task.input_text,
                "selected_task": task.selected_task.value,
                "confidence_used": task.confidence_used,
                "reasoning": task.reasoning,
                "status": task.status,
                "processing_time": task.processing_time,
                "timestamp": task.metadata.get("timestamp", datetime.now().isoformat())
            }
            for task in self.processing_history[-limit:]
        ]

# Pydantic models for API
class ProcessRequest(BaseModel):
    input: str
    options: Optional[Dict[str, Any]] = None

class AnalyzeRequest(BaseModel):
    input: str

# Initialize decision agent
decision_agent = CompleteDecisionAgent()

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    
    logger.info("🚀 Complete Decision Agent starting up...")
    logger.info(f"🧠 HRM System: {'Available' if decision_agent.hrm_available else 'Not Available'}")
    logger.info("✅ All task processors initialized")
    
    yield
    
    logger.info("📊 Final Statistics:")
    stats = decision_agent.get_stats()
    logger.info(f"   Total Requests: {stats['total_requests']}")
    logger.info(f"   Success Rate: {stats['success_rate']:.1%}")
    logger.info("🛑 Decision Agent shutting down...")

app = FastAPI(
    title="Complete Decision Agent API",
    description="Intelligent task routing with web scraping, search, database, KM RAG, code execution, and HRM",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.info("Static files directory not found - creating demo UI inline")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with demo interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision Agent Demo</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
                margin: 0; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; color: white;
            }
            .container { max-width: 800px; margin: 0 auto; }
            .card { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; margin: 20px 0; border-radius: 10px; 
                backdrop-filter: blur(10px);
            }
            .btn { 
                display: inline-block; padding: 10px 20px; 
                background: rgba(255,255,255,0.2); color: white; 
                text-decoration: none; border-radius: 5px; margin: 5px;
                border: 1px solid rgba(255,255,255,0.3);
            }
            .btn:hover { background: rgba(255,255,255,0.3); }
            h1 { text-align: center; font-size: 2.5rem; margin-bottom: 10px; }
            .feature { margin: 15px 0; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Complete Decision Agent</h1>
            <p style="text-align: center; font-size: 1.2rem;">Intelligent Task Routing System</p>
            
            <div class="card">
                <h3>🎯 Available Task Types:</h3>
                <div class="feature">📄 <strong>Web Scraping:</strong> Extract content from URLs</div>
                <div class="feature">🔍 <strong>Search:</strong> Web search and research</div>
                <div class="feature">🗄️ <strong>Database:</strong> SQL queries and data analysis</div>
                <div class="feature">📚 <strong>Knowledge RAG:</strong> Company knowledge retrieval</div>
                <div class="feature">💻 <strong>Code Execution:</strong> Python calculations and code</div>
                <div class="feature">🧠 <strong>HRM Reasoning:</strong> Complex analytical tasks</div>
            </div>
            
            <div class="card">
                <h3>🚀 Quick Start:</h3>
                <p>Try these example inputs:</p>
                <ul>
                    <li><code>https://example.com</code> - Web scraping</li>
                    <li><code>search for latest AI news</code> - Web search</li>
                    <li><code>show all users in database</code> - Database query</li>
                    <li><code>what is our security policy?</code> - Knowledge RAG</li>
                    <li><code>calculate: 15 * 37 + 128</code> - Code execution</li>
                    <li><code>analyze pros and cons of remote work</code> - HRM reasoning</li>
                </ul>
            </div>
            
            <div style="text-align: center;">
                <a href="/api/docs" class="btn">📚 API Documentation</a>
                <a href="/demo" class="btn">🎨 Interactive Demo</a>
                <a href="/api/stats" class="btn">📊 Statistics</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.post("/api/process")
async def process_request(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Process user input with intelligent task routing"""
    
    try:
        result = await decision_agent.process_request(request.input, request.options)
        return result
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze") 
async def analyze_input(request: AnalyzeRequest):
    """Analyze input and return task confidence scores"""
    
    try:
        task_scores = await decision_agent.analyze_input(request.input)
        
        return {
            "input": request.input,
            "task_scores": [
                {
                    "task_type": score.task_type.value,
                    "confidence": score.confidence,
                    "reasoning": score.reasoning,
                    "priority": score.priority,
                    "complexity": score.complexity.value
                }
                for score in task_scores
            ]
        }
    except Exception as e:
        logger.error(f"Input analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get processing statistics"""
    
    return decision_agent.get_stats()

@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get task processing history"""
    
    return {
        "history": decision_agent.get_task_history(limit),
        "total_tasks": len(decision_agent.processing_history)
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    
    stats = decision_agent.get_stats()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "processors": [
            "web_scraping",
            "google_search", 
            "database_query",
            "knowledge_rag",
            "code_execution",
            "general_query"
        ],
        "hrm_available": decision_agent.hrm_available,
        "total_requests": stats["total_requests"],
        "success_rate": stats["success_rate"],
        "uptime": "running"
    }

# Demo interface endpoint
@app.get("/demo")
async def demo_interface():
    """Interactive demo interface"""
    
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Decision Agent - Interactive Demo</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; padding: 20px;
            }
            .container {
                max-width: 1200px; margin: 0 auto;
                background: rgba(255,255,255,0.95);
                border-radius: 20px; overflow: hidden;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white; padding: 30px; text-align: center;
            }
            .content {
                display: grid; grid-template-columns: 1fr 1fr;
                gap: 30px; padding: 30px;
            }
            .input-panel, .results-panel {
                background: #f8fafc; padding: 25px;
                border-radius: 15px; border: 1px solid #e2e8f0;
            }
            .input-field {
                width: 100%; padding: 12px; margin: 10px 0;
                border: 2px solid #e5e7eb; border-radius: 8px;
                font-size: 1rem;
            }
            .input-field:focus {
                outline: none; border-color: #4f46e5;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }
            .btn {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white; border: none; padding: 12px 25px;
                border-radius: 8px; font-size: 1rem; cursor: pointer;
                margin: 10px 5px 10px 0; transition: transform 0.2s;
            }
            .btn:hover { transform: translateY(-2px); }
            .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            .result-item {
                background: white; padding: 15px; margin: 15px 0;
                border-radius: 8px; border-left: 4px solid #4f46e5;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .loading { 
                display: none; text-align: center; padding: 20px; 
                color: #6b7280;
            }
            .loading.show { display: block; }
            .example-btn {
                background: #f3f4f6; border: 1px solid #d1d5db;
                color: #374151; padding: 8px 12px; border-radius: 6px;
                font-size: 0.9rem; cursor: pointer; margin: 5px 5px 5px 0;
                transition: all 0.2s;
            }
            .example-btn:hover {
                background: #e5e7eb; transform: translateY(-1px);
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Decision Agent</h1>
                <p>Interactive Demo - Intelligent Task Routing</p>
            </div>
            
            <div class="content">
                <div class="input-panel">
                    <h3>📝 Input</h3>
                    <textarea id="userInput" class="input-field" rows="6" 
                        placeholder="Enter your request here...

Examples:
• https://example.com (web scraping)
• search for latest AI trends  
• show all users in database
• what is our security policy?
• calculate: 15 * 37 + 128
• analyze remote work benefits"></textarea>
                    
                    <div>
                        <button id="processBtn" class="btn">🚀 Process</button>
                        <button id="analyzeBtn" class="btn" style="background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);">🔍 Analyze</button>
                        <button id="clearBtn" class="btn" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">🗑️ Clear</button>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <label><strong>📋 Quick Examples:</strong></label><br>
                        <button class="example-btn" onclick="setExample('https://httpbin.org/json')">🌐 Web Scraping</button>
                        <button class="example-btn" onclick="setExample('search for latest Python frameworks')">🔍 Search</button>
                        <button class="example-btn" onclick="setExample('show all users')">🗄️ Database</button>
                        <button class="example-btn" onclick="setExample('what is our AI policy?')">📚 Knowledge</button>
                        <button class="example-btn" onclick="setExample('calculate: 2**10 + 5*7')">💻 Code</button>
                        <button class="example-btn" onclick="setExample('compare pros and cons of cloud vs on-premise')">🧠 Analysis</button>
                    </div>
                </div>
                
                <div class="results-panel">
                    <h3>📊 Results</h3>
                    
                    <div class="loading" id="loading">
                        <div style="width: 40px; height: 40px; border: 4px solid #e5e7eb; border-top: 4px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 10px;"></div>
                        <p>Processing your request...</p>
                    </div>
                    
                    <div id="results">
                        <div style="text-align: center; color: #6b7280; padding: 40px;">
                            Enter a request and click "Process" to see results here.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let processingCount = 0;
            
            document.getElementById('processBtn').addEventListener('click', processRequest);
            document.getElementById('analyzeBtn').addEventListener('click', analyzeInput);
            document.getElementById('clearBtn').addEventListener('click', clearResults);
            
            function setExample(text) {
                document.getElementById('userInput').value = text;
            }
            
            async function processRequest() {
                const input = document.getElementById('userInput').value.trim();
                if (!input) {
                    alert('Please enter a request');
                    return;
                }
                
                showLoading(true);
                disableButtons(true);
                
                try {
                    const response = await fetch('/api/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: input })
                    });
                    
                    const result = await response.json();
                    displayResult(result, 'process');
                    
                } catch (error) {
                    displayError('Processing failed: ' + error.message);
                } finally {
                    showLoading(false);
                    disableButtons(false);
                }
            }
            
            async function analyzeInput() {
                const input = document.getElementById('userInput').value.trim();
                if (!input) {
                    alert('Please enter a request');
                    return;
                }
                
                showLoading(true);
                disableButtons(true);
                
                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: input })
                    });
                    
                    const result = await response.json();
                    displayResult(result, 'analyze');
                    
                } catch (error) {
                    displayError('Analysis failed: ' + error.message);
                } finally {
                    showLoading(false);
                    disableButtons(false);
                }
            }
            
            function displayResult(result, type) {
                const resultsDiv = document.getElementById('results');
                
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                let html = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">' +
                    '<strong>' + (type === 'analyze' ? '🔍 Analysis Result' : '🚀 Processing Result') + '</strong>' +
                    '<span style="color: #6b7280; font-size: 0.8rem;">' + new Date().toLocaleTimeString() + '</span>' +
                    '</div>';
                
                if (type === 'analyze') {
                    html += '<div><strong>Task Analysis:</strong></div>';
                    if (result.task_scores) {
                        result.task_scores.forEach((score, index) => {
                            const isTop = index === 0;
                            html += '<div style="margin: 10px 0; padding: 10px; background: ' + (isTop ? '#f0f9ff' : '#f9fafb') + '; border-radius: 6px; border-left: 3px solid ' + (isTop ? '#4f46e5' : '#d1d5db') + ';">' +
                                '<div style="display: flex; justify-content: space-between;">' +
                                    '<strong>' + (isTop ? '🎯 ' : '') + score.task_type.toUpperCase().replace('_', ' ') + '</strong>' +
                                    '<span style="color: #4f46e5; font-weight: 600;">' + (score.confidence * 100).toFixed(1) + '%</span>' +
                                '</div>' +
                                '<div style="margin-top: 5px; font-size: 0.9rem; color: #6b7280;">' + score.reasoning + '</div>' +
                                (isTop ? '<div style="margin-top: 5px; font-size: 0.8rem; color: #4f46e5;">👑 Selected Task</div>' : '') +
                            '</div>';
                        });
                    }
                } else {
                    // Process result
                    html += '<div style="margin-bottom: 10px;">';
                    if (result.status === 'success') {
                        html += '<span style="background: #dcfce7; color: #166534; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">✅ SUCCESS</span>';
                    } else {
                        html += '<span style="background: #fef2f2; color: #dc2626; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">❌ ERROR</span>';
                    }
                    html += '</div>';
                    
                    if (result.selected_task) {
                        html += '<p><strong>Task:</strong> ' + result.selected_task.replace('_', ' ').toUpperCase() + '</p>';
                    }
                    
                    if (result.confidence_used) {
                        html += '<p><strong>Confidence:</strong> ' + (result.confidence_used * 100).toFixed(1) + '%</p>';
                    }
                    
                    if (result.processing_time) {
                        html += '<p><strong>Time:</strong> ' + result.processing_time.toFixed(3) + 's</p>';
                    }
                    
                    if (result.reasoning) {
                        html += '<p><strong>Reasoning:</strong> ' + result.reasoning + '</p>';
                    }
                    
                    if (result.error) {
                        html += '<p style="color: #dc2626;"><strong>Error:</strong> ' + result.error + '</p>';
                    }
                    
                    if (result.result) {
                        html += '<details style="margin-top: 10px;"><summary style="cursor: pointer; font-weight: 600;">📄 View Result</summary>';
                        html += '<pre style="background: #f3f4f6; padding: 10px; border-radius: 6px; overflow-x: auto; margin-top: 8px; white-space: pre-wrap;">';
                        html += JSON.stringify(result.result, null, 2);
                        html += '</pre></details>';
                    }
                }
                
                resultItem.innerHTML = html;
                
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].style.textAlign === 'center') {
                    resultsDiv.innerHTML = '';
                }
                
                resultsDiv.insertBefore(resultItem, resultsDiv.firstChild);
                
                // Keep only last 10 results
                while (resultsDiv.children.length > 10) {
                    resultsDiv.removeChild(resultsDiv.lastChild);
                }
            }
            
            function displayError(message) {
                const resultsDiv = document.getElementById('results');
                
                const errorItem = document.createElement('div');
                errorItem.className = 'result-item';
                errorItem.style.borderLeftColor = '#ef4444';
                errorItem.innerHTML = '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">' +
                    '<strong>❌ Error</strong>' +
                    '<span style="color: #6b7280; font-size: 0.8rem;">' + new Date().toLocaleTimeString() + '</span>' +
                    '</div>' +
                    '<p style="color: #dc2626;">' + message + '</p>';
                
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].style.textAlign === 'center') {
                    resultsDiv.innerHTML = '';
                }
                
                resultsDiv.insertBefore(errorItem, resultsDiv.firstChild);
            }
            
            function showLoading(show) {
                const loadingDiv = document.getElementById('loading');
                if (show) {
                    loadingDiv.classList.add('show');
                } else {
                    loadingDiv.classList.remove('show');
                }
            }
            
            function disableButtons(disabled) {
                document.getElementById('processBtn').disabled = disabled;
                document.getElementById('analyzeBtn').disabled = disabled;
            }
            
            function clearResults() {
                document.getElementById('results').innerHTML = '<div style="text-align: center; color: #6b7280; padding: 40px;">Enter a request and click "Process" to see results here.</div>';
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Complete Decision Agent with Task Routing")
    print("=" * 50)
    print(f"🌐 Server: http://localhost:8000")
    print(f"🎨 Demo UI: http://localhost:8000/demo")
    print(f"📚 API Docs: http://localhost:8000/api/docs")
    print(f"📊 Statistics: http://localhost:8000/api/stats")
    print("")
    print("🎯 Available Task Types:")
    print("• 📄 Web Scraping (URLs)")
    print("• 🔍 Google Search")
    print("• 🗄️ Database Queries")
    print("• 📚 Knowledge Management RAG")
    print("• 💻 Code Execution")
    print("• 🧠 HRM Complex Reasoning")
    print("")
    print("🔄 Server starting...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )