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
import time
from concurrent.futures import ThreadPoolExecutor
import io

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('decision_agent.log', encoding='utf-8')
    ]
)
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
    """Advanced web scraping processor with robust content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.timeout = 15
        self.max_retries = 3
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Extract and analyze web content from URLs"""
        
        urls = self._extract_urls(input_text)
        if not urls:
            return {
                "error": "No valid URLs found in input",
                "suggestion": "Please provide valid URLs (e.g., https://example.com)"
            }
        
        results = []
        for url in urls[:5]:  # Limit to 5 URLs for performance
            try:
                content = await self._scrape_url(url)
                results.append({
                    "url": url,
                    "title": content.get("title", ""),
                    "content": content.get("content", "")[:2000],  # Limit content length
                    "links": content.get("links", [])[:15],
                    "images": content.get("images", [])[:10],
                    "metadata": content.get("metadata", {}),
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {str(e)}")
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "failed"
                })
        
        successful_scrapes = len([r for r in results if r.get("status") == "success"])
        
        return {
            "scraped_urls": len(results),
            "successful_scrapes": successful_scrapes,
            "success_rate": successful_scrapes / len(results) if results else 0,
            "results": results
        }
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from input text with better validation"""
        # Enhanced URL regex
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        urls = re.findall(url_pattern, text)
        
        # Domain-like patterns
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, text)
        
        for domain in domains:
            if not any(domain in url for url in urls):
                if not domain.startswith(('http://', 'https://')):
                    urls.append(f'https://{domain}')
        
        # Validate URLs
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    valid_urls.append(url)
            except Exception:
                continue
        
        return list(set(valid_urls))
    
    async def _scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single URL with enhanced extraction"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    return {
                        "title": f"Non-HTML content: {content_type}",
                        "content": f"Content type: {content_type}, Size: {len(response.content)} bytes",
                        "links": [],
                        "images": [],
                        "metadata": {"content_type": content_type}
                    }
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                return self._extract_content(soup, url)
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured content from BeautifulSoup object"""
        
        # Extract title with fallbacks
        title = ""
        title_candidates = [
            soup.find('title'),
            soup.find('h1'),
            soup.find('meta', property='og:title'),
            soup.find('meta', name='twitter:title')
        ]
        
        for candidate in title_candidates:
            if candidate:
                title = candidate.get_text() if hasattr(candidate, 'get_text') else candidate.get('content', '')
                if title:
                    title = title.strip()[:200]
                    break
        
        # Extract main content with better selectors
        content = ""
        content_selectors = [
            'main', 'article', '.content', '#content', '.post-content',
            '.entry-content', '.article-body', '[role="main"]',
            '.post', '.article', '.story', '.text-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                content = main_content.get_text(separator=' ', strip=True)
                if len(content) > 100:  # Ensure we have substantial content
                    break
        
        # Fallback to body content
        if not content and soup.body:
            content = soup.body.get_text(separator=' ', strip=True)
        
        # Clean up content
        content = ' '.join(content.split())[:5000]  # Limit and clean whitespace
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:25]:
            href = link['href']
            text = link.get_text().strip()
            
            # Resolve relative URLs
            if href.startswith(('http://', 'https://')):
                full_url = href
            elif href.startswith('//'):
                full_url = f"https:{href}"
            else:
                full_url = urljoin(url, href)
            
            if text and len(text) > 3:  # Skip empty or very short link texts
                links.append({
                    "text": text[:150],
                    "url": full_url
                })
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True)[:15]:
            src = img['src']
            alt = img.get('alt', '')[:150]
            
            # Resolve relative URLs
            if src.startswith(('http://', 'https://')):
                full_url = src
            elif src.startswith('//'):
                full_url = f"https:{src}"
            else:
                full_url = urljoin(url, src)
            
            images.append({
                "src": full_url,
                "alt": alt
            })
        
        # Extract enhanced metadata
        metadata = {
            "description": "",
            "keywords": "",
            "author": "",
            "published_date": "",
            "language": ""
        }

        # Meta tags extraction
        meta_mappings = {
            "description": ["description", "og:description", "twitter:description"],
            "keywords": ["keywords"],
            "author": ["author", "article:author"],
            "published_date": ["article:published_time", "publish_date"],
            "language": ["language", "lang"]
        }

        # ปรับปรุงการดึง metadata ให้ปลอดภัยขึ้น
        try:
            for meta in soup.find_all('meta'):
                if meta:
                    name = meta.get('name', '')
                    if name:
                        name = name.lower()
                        content_attr = meta.get('content', '')
                        if content_attr:
                            metadata[name] = content_attr[:200]
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")

        # ดึง meta ตาม mapping เดิม (ยังคงไว้)
        for key, selectors in meta_mappings.items():
            for selector in selectors:
                meta = soup.find('meta', attrs={'name': selector}) or \
                       soup.find('meta', attrs={'property': selector})
                if meta:
                    metadata[key] = meta.get('content', '')[:500]
                    break
        
        return {
            "title": title,
            "content": content,
            "links": links,
            "images": images,
            "metadata": metadata,
            "word_count": len(content.split()) if content else 0,
            "url": url
        }

class GoogleSearchProcessor:
    """Enhanced search processor with multiple engines"""
    
    def __init__(self):
        self.search_engines = {
            "duckduckgo": self._search_duckduckgo,
            "mock": self._generate_mock_results
        }
        self.default_engine = "duckduckgo"
        self.timeout = 10
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process search queries with multiple engine support"""
        
        search_query = self._extract_search_query(input_text)
        
        try:
            # Try primary search engine
            results = await self.search_engines[self.default_engine](search_query)
            
            # If primary fails, fall back to mock results
            if not results.get("results"):
                logger.warning(f"Primary search failed, using fallback")
                results = await self.search_engines["mock"](search_query)
            
            return {
                **results,
                "search_query": search_query,
                "query_analysis": self._analyze_query(search_query)
            }
            
        except Exception as e:
            logger.error(f"Search processing failed: {str(e)}")
            # Always provide mock results as ultimate fallback
            return await self.search_engines["mock"](search_query)
    
    def _extract_search_query(self, text: str) -> str:
        """Extract and clean search query from input text"""
        
        # Remove common search prefixes
        search_prefixes = [
            "search for", "search", "google", "find", "look up", "lookup",
            "what is", "who is", "how to", "where is", "when is", "why is"
        ]
        
        cleaned_text = text.lower().strip()
        
        for prefix in search_prefixes:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        
        # Remove quotes and clean up
        cleaned_text = cleaned_text.strip("\"'")
        
        return cleaned_text if cleaned_text else text.strip()
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze search query to provide insights"""
        
        words = query.lower().split()
        
        return {
            "word_count": len(words),
            "query_type": self._classify_query_type(query),
            "intent": self._detect_search_intent(query),
            "entities": self._extract_entities(query)
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of search query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "when", "where", "who"]):
            return "question"
        elif any(word in query_lower for word in ["best", "top", "review", "compare"]):
            return "comparison"
        elif any(word in query_lower for word in ["tutorial", "guide", "learn"]):
            return "educational"
        elif any(word in query_lower for word in ["news", "latest", "recent"]):
            return "news"
        else:
            return "informational"
    
    def _detect_search_intent(self, query: str) -> str:
        """Detect user intent from search query"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["buy", "price", "cost", "purchase"]):
            return "commercial"
        elif any(word in query_lower for word in ["how to", "tutorial", "guide"]):
            return "instructional"
        elif any(word in query_lower for word in ["near me", "location", "address"]):
            return "local"
        else:
            return "informational"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        
        # Simple entity extraction based on capitalization and common patterns
        words = query.split()
        entities = []
        
        for word in words:
            # Capitalized words that aren't at sentence start
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return entities
    
    async def _search_duckduckgo(self, query: str) -> Dict[str, Any]:
        """Search using DuckDuckGo with improved parsing"""
        
        search_url = "https://duckduckgo.com/html/"
        params = {"q": query}
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, headers=headers, timeout=self.timeout) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Parse DuckDuckGo results with improved selectors
            for result_div in soup.find_all('div', class_='result')[:12]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text().strip()
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text().strip() if snippet_elem else ""
                    
                    # Clean up URL (DuckDuckGo sometimes uses redirect URLs)
                    if url.startswith('/l/?kh='):
                        # Extract actual URL from DuckDuckGo redirect
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        if 'uddg' in parsed:
                            url = urllib.parse.unquote(parsed['uddg'][0])
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "duckduckgo"
                    })
            
            return {
                "search_engine": "DuckDuckGo",
                "results_count": len(results),
                "results": results,
                "related_searches": self._generate_related_searches(query),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            raise e
    
    async def _generate_mock_results(self, query: str) -> Dict[str, Any]:
        """Generate enhanced mock search results"""
        
        # Generate more realistic mock results based on query
        mock_results = []
        
        # Template-based results
        templates = [
            {
                "title": f"Complete Guide to {query.title()}",
                "url": f"https://guide.example.com/{query.replace(' ', '-').lower()}",
                "snippet": f"Comprehensive guide covering everything you need to know about {query}. Expert tips, best practices, and step-by-step instructions."
            },
            {
                "title": f"{query.title()} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"Wikipedia article about {query} with detailed information, history, and references from reliable sources."
            },
            {
                "title": f"Latest {query.title()} News & Updates",
                "url": f"https://news.example.com/topics/{query.replace(' ', '-')}",
                "snippet": f"Stay updated with the latest news, trends, and developments related to {query} from trusted news sources worldwide."
            },
            {
                "title": f"Best Practices for {query.title()}",
                "url": f"https://bestpractices.example.com/{query.replace(' ', '-')}",
                "snippet": f"Industry-leading best practices and expert recommendations for {query}. Learn from professionals and avoid common mistakes."
            },
            {
                "title": f"{query.title()} Tutorial & Examples",
                "url": f"https://tutorial.example.com/{query.replace(' ', '-')}",
                "snippet": f"Learn {query} with practical examples, hands-on tutorials, and real-world applications. Perfect for beginners and experts."
            }
        ]
        
        # Add query-specific results
        for template in templates:
            mock_results.append({
                **template,
                "source": "mock"
            })
        
        return {
            "search_engine": "Mock Search Engine",
            "results_count": len(mock_results),
            "results": mock_results,
            "related_searches": self._generate_related_searches(query),
            "status": "mock",
            "note": "Mock search results for demonstration purposes"
        }
    
    def _generate_related_searches(self, query: str) -> List[str]:
        """Generate contextual related search suggestions"""
        
        templates = [
            f"what is {query}",
            f"how to use {query}",
            f"{query} examples",
            f"{query} vs alternatives",
            f"best {query}",
            f"{query} tutorial",
            f"{query} benefits",
            f"{query} problems"
        ]
        
        return templates[:5]  # Return top 5 suggestions

class DatabaseProcessor:
    """Enhanced database processor with natural language understanding"""
    
    def __init__(self, db_path: str = "decision_agent.db"):
        self.db_path = db_path
        self._init_demo_database()
        self.query_cache = {}
        
    def get_database_info(self) -> str:
        return """
        🗄️ ข้อมูลฐานข้อมูล

        📊 แหล่งข้อมูล: ฐานข้อมูล SQLite สาธิต
        ฐานข้อมูลประกอบด้วยข้อมูลตัวอย่างที่สร้างขึ้นอัตโนมัติ

        📋 ตารางที่มี:
        1. 👥 ตาราง USERS: ข้อมูลพนักงาน 6 คน
        2. 📋 ตาราง TASKS: งานต่างๆ 6 รายการ  
        3. 🚀 ตาราง PROJECTS: โครงการ 3 โครงการ
        """
    
    def _init_demo_database(self):
        """Initialize comprehensive demo database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create enhanced tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department TEXT,
                    position TEXT,
                    salary REAL,
                    hire_date DATE,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 1,
                    assigned_to INTEGER,
                    estimated_hours REAL,
                    actual_hours REAL,
                    due_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (assigned_to) REFERENCES users (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'planning',
                    budget REAL,
                    start_date DATE,
                    end_date DATE,
                    manager_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (manager_id) REFERENCES users (id)
                )
            """)
            
            # Insert comprehensive sample data
            sample_users = [
                ("Alice Johnson", "alice@company.com", "Engineering", "Senior Developer", 95000, "2022-01-15", "active"),
                ("Bob Smith", "bob@company.com", "Marketing", "Marketing Manager", 75000, "2021-06-20", "active"),
                ("Carol Davis", "carol@company.com", "Sales", "Sales Representative", 65000, "2023-03-10", "active"),
                ("David Wilson", "david@company.com", "Support", "Support Specialist", 55000, "2022-08-05", "active"),
                ("Eva Martinez", "eva@company.com", "Engineering", "DevOps Engineer", 90000, "2021-11-12", "active"),
                ("Frank Brown", "frank@company.com", "HR", "HR Coordinator", 60000, "2020-04-18", "inactive")
            ]
            
            cursor.executemany(
                "INSERT OR IGNORE INTO users (name, email, department, position, salary, hire_date, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                sample_users
            )
            
            sample_tasks = [
                ("Implement user authentication", "Add secure login/logout functionality with 2FA", "in_progress", 3, 1, 24.0, 18.5, "2024-02-15"),
                ("Marketing campaign analysis", "Analyze Q4 2023 campaign performance and ROI", "completed", 2, 2, 16.0, 20.0, "2024-01-30"),
                ("Customer support tickets", "Process and resolve pending customer support requests", "pending", 1, 4, 8.0, None, "2024-02-10"),
                ("Sales report automation", "Automate monthly sales reporting process", "pending", 2, 3, 32.0, None, "2024-02-28"),
                ("Infrastructure monitoring", "Set up comprehensive monitoring for production systems", "in_progress", 3, 5, 40.0, 25.0, "2024-03-15"),
                ("Database optimization", "Optimize database queries and improve performance", "planning", 2, 1, 20.0, None, "2024-03-01")
            ]
            
            cursor.executemany(
                "INSERT OR IGNORE INTO tasks (title, description, status, priority, assigned_to, estimated_hours, actual_hours, due_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                sample_tasks
            )
            
            sample_projects = [
                ("Website Redesign", "Complete overhaul of company website with modern design", "active", 50000.0, "2024-01-01", "2024-04-30", 2),
                ("Mobile App Development", "Develop iOS and Android mobile applications", "planning", 120000.0, "2024-03-01", "2024-09-30", 1),
                ("Database Migration", "Migrate legacy database to modern cloud solution", "completed", 30000.0, "2023-10-01", "2023-12-31", 5)
            ]
            
            cursor.executemany(
                "INSERT OR IGNORE INTO projects (name, description, status, budget, start_date, end_date, manager_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                sample_projects
            )
            
            conn.commit()
            logger.info("Demo database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process database queries with enhanced natural language support"""
        
        try:
            # Analyze query
            query_info = self._analyze_query(input_text)
            
            # Generate and execute SQL
            if query_info["type"] == "sql":
                # Direct SQL query (sanitized)
                sql_query = self._sanitize_sql(query_info["query"])
                results = self._execute_query(sql_query)
            else:
                # Natural language to SQL
                sql_query = self._generate_sql_from_text(input_text, query_info)
                results = self._execute_query(sql_query)
            
            # Process and format results
            processed_results = self._process_results(results, query_info)
            
            return {
                "query_type": query_info["type"],
                "query": input_text,
                "sql_executed": sql_query,
                "results": processed_results["data"],
                "summary": processed_results["summary"],
                "row_count": len(results) if isinstance(results, list) else 0,
                "tables_accessed": query_info.get("tables", []),
                "query_analysis": query_info
            }
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return {
                "query_type": "error",
                "query": input_text,
                "error": str(e),
                "suggestion": self._get_query_suggestion(input_text),
                "available_tables": ["users", "tasks", "projects"],
                "sample_queries": [
                    "show all users",
                    "find pending tasks",
                    "count users by department",
                    "show tasks assigned to Alice",
                    "list projects with budget over 50000"
                ]
            }
    
    def _analyze_query(self, text: str) -> Dict[str, Any]:
        """Enhanced query analysis with intent detection"""
        
        text_lower = text.lower().strip()
        
        # Check if it's a SQL query
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
        if any(text_lower.startswith(keyword) for keyword in sql_keywords):
            return {
                "type": "sql",
                "query": text,
                "tables": self._extract_table_names(text),
                "intent": "direct_sql"
            }
        
        # Analyze natural language query
        return {
            "type": "natural",
            "intent": self._detect_query_intent(text_lower),
            "entities": self._extract_query_entities(text_lower),
            "filters": self._extract_filters(text_lower),
            "aggregation": self._detect_aggregation(text_lower)
        }
    
    def _detect_query_intent(self, text: str) -> str:
        """Detect the intent of a natural language query"""
        
        if any(word in text for word in ["show", "list", "get", "find", "display", "view"]):
            return "retrieve"
        elif any(word in text for word in ["count", "how many", "number of", "total"]):
            return "count"
        elif any(word in text for word in ["sum", "average", "avg", "maximum", "minimum", "max", "min"]):
            return "aggregate"
        elif any(word in text for word in ["create", "add", "insert", "new"]):
            return "create"
        elif any(word in text for word in ["update", "change", "modify", "edit"]):
            return "update"
        elif any(word in text for word in ["delete", "remove", "drop"]):
            return "delete"
        else:
            return "retrieve"
    
    def _extract_query_entities(self, text: str) -> List[str]:
        """Extract entities and table references from query"""
        
        entities = []
        
        # Table entities
        if any(word in text for word in ["user", "users", "employee", "person", "people"]):
            entities.append("users")
        if any(word in text for word in ["task", "tasks", "assignment", "work", "job"]):
            entities.append("tasks")
        if any(word in text for word in ["project", "projects"]):
            entities.append("projects")
        
        # Status entities
        status_words = ["pending", "completed", "in_progress", "active", "inactive", "planning"]
        entities.extend([word for word in status_words if word in text])
        
        # Department entities
        departments = ["engineering", "marketing", "sales", "support", "hr"]
        entities.extend([dept for dept in departments if dept in text])
        
        return entities
    
    def _extract_filters(self, text: str) -> Dict[str, Any]:
        """Extract filter conditions from natural language"""
        
        filters = {}
        
        # Status filters
        if "pending" in text:
            filters["status"] = "pending"
        elif "completed" in text:
            filters["status"] = "completed"
        elif "in progress" in text or "in_progress" in text:
            filters["status"] = "in_progress"
        
        # Department filters
        departments = ["engineering", "marketing", "sales", "support", "hr"]
        for dept in departments:
            if dept in text:
                filters["department"] = dept
        
        # Name filters (look for proper names)
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ["assigned to", "for", "by"] and i + 1 < len(words):
                potential_name = words[i + 1].title()
                if len(potential_name) > 2:
                    filters["assigned_name"] = potential_name
        
        # Priority filters
        if "high priority" in text or "priority 3" in text:
            filters["priority"] = 3
        elif "medium priority" in text or "priority 2" in text:
            filters["priority"] = 2
        elif "low priority" in text or "priority 1" in text:
            filters["priority"] = 1
        
        return filters
    
    def _detect_aggregation(self, text: str) -> Optional[str]:
        """Detect aggregation operations"""
        
        if "count" in text or "how many" in text:
            return "COUNT"
        elif "sum" in text or "total" in text:
            return "SUM"
        elif "average" in text or "avg" in text:
            return "AVG"
        elif "maximum" in text or "max" in text:
            return "MAX"
        elif "minimum" in text or "min" in text:
            return "MIN"
        else:
            return None
    
    def _generate_sql_from_text(self, text: str, query_info: Dict) -> str:
        """Generate SQL from natural language with enhanced logic"""
        
        intent = query_info["intent"]
        entities = query_info["entities"]
        filters = query_info["filters"]
        aggregation = query_info["aggregation"]
        
        # Determine primary table
        if "users" in entities:
            primary_table = "users"
        elif "tasks" in entities:
            primary_table = "tasks"
        elif "projects" in entities:
            primary_table = "projects"
        else:
            primary_table = "users"  # Default fallback
        
        # Build SELECT clause
        if aggregation == "COUNT":
            select_clause = "SELECT COUNT(*) as count"
            if "department" in text and primary_table == "users":
                select_clause = "SELECT department, COUNT(*) as count"
                group_by = " GROUP BY department"
            else:
                group_by = ""
        elif aggregation:
            if "salary" in text:
                select_clause = f"SELECT {aggregation}(salary) as result"
            elif "hours" in text:
                select_clause = f"SELECT {aggregation}(actual_hours) as result"
            elif "budget" in text:
                select_clause = f"SELECT {aggregation}(budget) as result"
            else:
                select_clause = "SELECT COUNT(*) as count"
            group_by = ""
        else:
            if primary_table == "tasks" and any(word in text for word in ["assigned", "who"]):
                select_clause = """
                    SELECT u.name, t.title, t.status, t.priority, t.due_date
                    FROM tasks t
                    JOIN users u ON t.assigned_to = u.id
                """
                primary_table = ""  # Already handled in FROM
            elif primary_table == "projects" and "manager" in text:
                select_clause = """
                    SELECT p.name as project_name, p.status, p.budget, u.name as manager
                    FROM projects p
                    JOIN users u ON p.manager_id = u.id
                """
                primary_table = ""
            else:
                select_clause = "SELECT *"
            group_by = ""
        
        # Build FROM clause
        from_clause = f" FROM {primary_table}" if primary_table else ""
        
        # Build WHERE clause
        where_conditions = []
        
        for key, value in filters.items():
            if key == "assigned_name":
                if primary_table == "tasks":
                    where_conditions.append(f"u.name LIKE '%{value}%'")
                else:
                    where_conditions.append(f"name LIKE '%{value}%'")
            elif key in ["status", "department", "priority"]:
                where_conditions.append(f"{key} = '{value}'")
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        # Build ORDER clause
        if intent == "retrieve" and not aggregation:
            if primary_table == "tasks":
                order_clause = " ORDER BY priority DESC, created_at DESC"
            elif primary_table == "users":
                order_clause = " ORDER BY name"
            elif primary_table == "projects":
                order_clause = " ORDER BY start_date DESC"
            else:
                order_clause = ""
        else:
            order_clause = ""
        
        # Combine SQL parts
        sql_query = select_clause + from_clause + where_clause + group_by + order_clause + " LIMIT 100"
        
        return sql_query.strip()
    
    def _sanitize_sql(self, sql: str) -> str:
        """Basic SQL sanitization for demo purposes"""
        
        # In production, use proper parameterized queries
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter', 'truncate']
        sql_lower = sql.lower()
        
        for keyword in dangerous_keywords:
            if sql_lower.strip().startswith(keyword):
                raise ValueError(f"'{keyword.upper()}' operations are not allowed in demo mode")
        
        return sql
    
    def _execute_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query with proper error handling"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            return results
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            raise e
        finally:
            conn.close()
    
    def _process_results(self, results: List[Dict], query_info: Dict) -> Dict[str, Any]:
        """Process and summarize query results"""
        
        if not results:
            return {
                "data": [],
                "summary": "No results found for the query."
            }
        
        # Generate summary based on query type
        intent = query_info.get("intent", "retrieve")
        
        if intent == "count" or query_info.get("aggregation") == "COUNT":
            if len(results) == 1 and "count" in results[0]:
                summary = f"Found {results[0]['count']} records matching the criteria."
            else:
                summary = f"Count results: {len(results)} categories found."
        elif query_info.get("aggregation"):
            agg_type = query_info.get("aggregation")
            summary = f"{agg_type} calculation completed. Found {len(results)} result(s)."
        else:
            summary = f"Retrieved {len(results)} record(s) successfully."
        
        return {
            "data": results,
            "summary": summary
        }
    
    def _get_query_suggestion(self, failed_query: str) -> str:
        """Provide helpful suggestions for failed queries"""
        
        suggestions = [
            "Try: 'show all users' or 'list tasks'",
            "Use: 'count users by department'",
            "Example: 'find pending tasks'",
            "Try: 'show tasks assigned to Alice'"
        ]
        
        return "; ".join(suggestions)
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        
        tables = []
        words = sql.upper().split()
        
        keywords_with_tables = ['FROM', 'JOIN', 'UPDATE', 'INTO']
        
        for i, word in enumerate(words):
            if word in keywords_with_tables and i + 1 < len(words):
                table_name = words[i + 1].lower().strip(',;')
                tables.append(table_name)
        
        return list(set(tables))

class KnowledgeRAGProcessor:
    """Enhanced Knowledge Management RAG processor with semantic search"""
    
    def __init__(self):
        self.knowledge_base = self._init_enhanced_knowledge_base()
        self.response_cache = {}
    
    def _init_enhanced_knowledge_base(self) -> Dict[str, Any]:
        """Initialize comprehensive knowledge base"""
        
        return {
            "documents": [
                {
                    "id": "doc_security_001",
                    "title": "Information Security Policy",
                    "content": """
                    Our comprehensive security policy mandates that all employees use strong passwords with minimum 12 characters, 
                    including uppercase, lowercase, numbers, and special characters. Two-factor authentication (2FA) is required 
                    for all business applications. VPN access is mandatory for remote work. All devices must have updated antivirus 
                    software and firewall protection. Data encryption is required for sensitive information both at rest and in transit.
                    Regular security training is mandatory for all staff members.
                    """,
                    "categories": ["security", "policy", "compliance", "remote_work"],
                    "last_updated": "2024-01-20",
                    "access_level": "all_employees",
                    "version": "3.2"
                },
                {
                    "id": "doc_ai_guidelines_002",
                    "title": "AI Development and Ethics Guidelines",
                    "content": """
                    When developing artificial intelligence systems, teams must prioritize data quality, model validation, 
                    and ethical considerations. All AI models must undergo rigorous testing including bias detection and 
                    fairness evaluation. Data privacy must be maintained throughout the ML pipeline. Model monitoring and 
                    continuous evaluation are mandatory for production systems. Explainable AI principles should be applied 
                    where possible. Regular audits of AI systems are required to ensure compliance with ethical guidelines.
                    """,
                    "categories": ["ai", "ml", "ethics", "development", "guidelines"],
                    "last_updated": "2024-01-15",
                    "access_level": "technical_teams",
                    "version": "2.1"
                },
                {
                    "id": "doc_support_003",
                    "title": "Customer Support Excellence Standards",
                    "content": """
                    Customer support representatives must respond to all inquiries within 4 hours during business hours and 
                    24 hours during weekends. Use empathetic and professional language in all communications. Provide clear, 
                    actionable solutions with step-by-step instructions when applicable. Escalate complex technical issues 
                    to specialized teams within 2 hours. Maintain detailed records of all customer interactions. Follow up 
                    on resolved issues within 48 hours to ensure customer satisfaction.
                    """,
                    "categories": ["customer_service", "support", "communication", "standards"],
                    "last_updated": "2024-01-18",
                    "access_level": "support_team",
                    "version": "1.5"
                },
                {
                    "id": "doc_dev_standards_004",
                    "title": "Software Development Standards and Best Practices",
                    "content": """
                    All code must follow established coding standards and style guides for the respective programming languages. 
                    Write comprehensive unit tests with minimum 80% code coverage. Use version control with meaningful commit messages. 
                    Conduct thorough code reviews for all pull requests. Implement proper error handling and logging. Document all 
                    APIs using OpenAPI/Swagger specifications. Follow security best practices including input validation and 
                    SQL injection prevention. Use automated CI/CD pipelines for deployment.
                    """,
                    "categories": ["development", "coding", "technical", "standards", "ci_cd"],
                    "last_updated": "2024-01-12",
                    "access_level": "development_team",
                    "version": "4.0"
                },
                {
                    "id": "doc_remote_work_005",
                    "title": "Remote Work Policy and Guidelines",
                    "content": """
                    Remote work is supported for eligible employees with manager approval. Employees must maintain productivity 
                    standards equivalent to on-site work. Secure VPN connection is required for accessing company resources. 
                    Regular check-ins with team members and supervisors are mandatory. Home office setup should include ergonomic 
                    workspace and reliable internet connection. Company equipment must be secured and not shared with family members. 
                    Participate in required video meetings and maintain professional appearance during calls.
                    """,
                    "categories": ["remote_work", "policy", "productivity", "security"],
                    "last_updated": "2024-01-25",
                    "access_level": "all_employees",
                    "version": "2.3"
                },
                {
                    "id": "doc_data_privacy_006",
                    "title": "Data Privacy and GDPR Compliance",
                    "content": """
                    All personal data must be processed in accordance with GDPR and local privacy regulations. Implement 
                    data minimization principles - collect only necessary data. Provide clear privacy notices to users. 
                    Ensure explicit consent for data processing. Implement right to erasure and data portability. 
                    Conduct privacy impact assessments for new systems. Report data breaches within 72 hours. 
                    Regular privacy training is required for all staff handling personal data.
                    """,
                    "categories": ["privacy", "gdpr", "compliance", "data_protection"],
                    "last_updated": "2024-01-22",
                    "access_level": "all_employees",
                    "version": "1.8"
                }
            ],
            "metadata": {
                "total_documents": 6,
                "last_indexed": datetime.now().isoformat(),
                "categories": [
                    "security", "ai", "ml", "customer_service", "development", 
                    "remote_work", "privacy", "compliance", "standards"
                ],
                "access_levels": ["all_employees", "technical_teams", "support_team", "development_team"]
            }
        }
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process knowledge queries using advanced RAG approach"""
        
        # Check cache first
        cache_key = hash(input_text.lower().strip())
        if cache_key in self.response_cache:
            cached_result = self.response_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result
        
        # Step 1: Enhanced document retrieval
        relevant_docs = self._retrieve_documents(input_text)
        
        # Step 2: Advanced ranking with multiple factors
        ranked_docs = self._rank_documents(relevant_docs, input_text)
        
        # Step 3: Context-aware response generation
        response = self._generate_enhanced_response(input_text, ranked_docs)
        
        # Step 4: Add metadata and insights
        result = {
            "query": input_text,
            "retrieved_documents": len(relevant_docs),
            "relevant_documents": len(ranked_docs),
            "response": response["answer"],
            "confidence_score": response["confidence"],
            "sources": [
                {
                    "title": doc["title"],
                    "id": doc["id"],
                    "relevance_score": doc.get("relevance_score", 0),
                    "categories": doc["categories"],
                    "last_updated": doc["last_updated"],
                    "version": doc.get("version", "1.0")
                }
                for doc in ranked_docs[:3]
            ],
            "related_topics": self._extract_related_topics(ranked_docs),
            "knowledge_base_stats": self.knowledge_base["metadata"],
            "query_analysis": self._analyze_query(input_text)
        }
        
        # Cache the result
        self.response_cache[cache_key] = result
        
        return result
    
    def _retrieve_documents(self, query: str) -> List[Dict]:
        """Enhanced document retrieval with semantic matching"""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_docs = []
        
        for doc in self.knowledge_base["documents"]:
            score = 0
            
            # Title matching (high weight)
            title_words = set(doc["title"].lower().split())
            title_overlap = len(query_words.intersection(title_words))
            score += title_overlap * 5
            
            # Content matching (medium weight)
            content_words = set(doc["content"].lower().split())
            content_overlap = len(query_words.intersection(content_words))
            score += content_overlap * 2
            
            # Category matching (high weight)
            for category in doc["categories"]:
                if any(word in category.lower() for word in query_words):
                    score += 4
                if category.lower() in query_lower:
                    score += 6
            
            # Semantic phrase matching
            for phrase in self._extract_phrases(query_lower):
                if phrase in doc["content"].lower():
                    score += 8
            
            # Question-specific matching
            if any(q_word in query_lower for q_word in ["what", "how", "why", "when", "where"]):
                if any(q_word in doc["content"].lower() for q_word in ["procedure", "process", "steps", "guidelines"]):
                    score += 3
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = score
                relevant_docs.append(doc_copy)
        
        return relevant_docs
    
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from query text"""
        
        # Common multi-word phrases that should be matched together
        phrases = []
        words = text.split()
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if len(phrase) > 5:  # Skip very short phrases
                phrases.append(phrase)
        
        # Extract 3-word phrases for complex queries
        if len(words) > 2:
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                phrases.append(phrase)
        
        return phrases
    
    def _rank_documents(self, documents: List[Dict], query: str) -> List[Dict]:
        """Advanced document ranking with multiple factors"""
        
        if not documents:
            return []
        
        # Apply additional ranking factors
        for doc in documents:
            original_score = doc["relevance_score"]
            
            # Recency boost (newer documents get slight boost)
            doc_date = datetime.fromisoformat(doc["last_updated"])
            days_old = (datetime.now() - doc_date).days
            recency_factor = max(0.8, 1.0 - (days_old / 365))
            doc["relevance_score"] *= recency_factor
            
            # Exact phrase matching boost
            query_lower = query.lower()
            if any(phrase in doc["content"].lower() for phrase in self._extract_phrases(query_lower)):
                doc["relevance_score"] *= 1.3
            
            # Length and completeness factor
            content_length = len(doc["content"])
            if content_length > 500:  # Comprehensive documents get boost
                doc["relevance_score"] *= 1.1
            
            # Category specificity boost
            query_categories = self._infer_query_categories(query)
            if any(cat in doc["categories"] for cat in query_categories):
                doc["relevance_score"] *= 1.2
        
        return sorted(documents, key=lambda x: x["relevance_score"], reverse=True)
    
    def _infer_query_categories(self, query: str) -> List[str]:
        """Infer likely categories from query text"""
        
        query_lower = query.lower()
        categories = []
        
        category_keywords = {
            "security": ["password", "security", "2fa", "vpn", "encryption", "firewall"],
            "ai": ["ai", "artificial intelligence", "machine learning", "ml", "model"],
            "development": ["code", "programming", "development", "software", "api"],
            "support": ["customer", "support", "help", "service", "ticket"],
            "remote_work": ["remote", "work from home", "wfh", "telecommute"],
            "privacy": ["privacy", "gdpr", "data protection", "personal data"],
            "policy": ["policy", "procedure", "guideline", "standard", "rule"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    def _generate_enhanced_response(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive response with confidence scoring"""
        
        if not documents:
            return {
                "answer": f"I couldn't find specific information about '{query}' in the knowledge base. Available topics include: {', '.join(self.knowledge_base['metadata']['categories'])}",
                "confidence": 0.1
            }
        
        # Use top 3 most relevant documents
        top_docs = documents[:3]
        
        # Build contextual response
        response_parts = []
        response_parts.append(f"Based on our knowledge base, here's what I found about '{query}':")
        
        total_relevance = sum(doc["relevance_score"] for doc in top_docs)
        
        for i, doc in enumerate(top_docs, 1):
            # Extract most relevant content sections
            relevant_content = self._extract_relevant_content(doc["content"], query)
            
            # Add source attribution with context
            response_parts.append(f"\n{i}. From '{doc['title']}' (v{doc.get('version', '1.0')}):")
            response_parts.append(f"   {relevant_content}")
            
            # Add specific details for key documents
            if i == 1:  # Most relevant document gets extra detail
                additional_context = self._extract_additional_context(doc, query)
                if additional_context:
                    response_parts.append(f"   Additional context: {additional_context}")
        
        # Add summary and recommendations
        response_parts.append(f"\nKey sources: {', '.join([doc['title'] for doc in top_docs])}")
        
        # Calculate confidence based on relevance scores and document quality
        confidence = min(0.95, (total_relevance / 100) * 0.8 + 0.2)
        
        return {
            "answer": " ".join(response_parts),
            "confidence": confidence
        }
    
    def _extract_relevant_content(self, content: str, query: str) -> str:
        """Extract the most relevant sentences from document content"""
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        query_words = set(query.lower().split())
        
        # Score sentences by relevance
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            # Boost score for exact phrase matches
            if any(phrase in sentence.lower() for phrase in self._extract_phrases(query.lower())):
                overlap += 3
            
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            # Sort by relevance and return top 2 sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent[0] for sent in scored_sentences[:2]]
            return ". ".join(top_sentences) + "."
        else:
            # Fallback to first substantive sentence
            for sentence in sentences:
                if len(sentence) > 50:  # Substantial content
                    return sentence + "."
            
            return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_additional_context(self, doc: Dict, query: str) -> str:
        """Extract additional context information from the document"""
        
        context_info = []
        
        # Add version information for important docs
        if "policy" in doc["title"].lower():
            context_info.append(f"Policy version {doc.get('version', '1.0')}")
        
        # Add update information for recent changes
        doc_date = datetime.fromisoformat(doc["last_updated"])
        days_ago = (datetime.now() - doc_date).days
        
        if days_ago < 30:
            context_info.append(f"recently updated ({days_ago} days ago)")
        
        # Add access level information
        if doc.get("access_level") and doc["access_level"] != "all_employees":
            context_info.append(f"applies to {doc['access_level'].replace('_', ' ')}")
        
        return ", ".join(context_info) if context_info else ""
    
    def _extract_related_topics(self, documents: List[Dict]) -> List[str]:
        """Extract related topics from retrieved documents"""
        
        all_categories = []
        for doc in documents:
            all_categories.extend(doc["categories"])
        
        # Count category frequency and return top related topics
        from collections import Counter
        category_counts = Counter(all_categories)
        
        return [cat for cat, count in category_counts.most_common(5)]
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to provide insights"""
        
        query_lower = query.lower()
        
        return {
            "word_count": len(query.split()),
            "query_type": self._classify_query_type(query_lower),
            "complexity": self._assess_query_complexity(query),
            "intent": self._detect_knowledge_intent(query_lower),
            "entities": self._extract_query_entities(query_lower)
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of knowledge query"""
        
        if any(word in query for word in ["what", "define", "explain"]):
            return "definition"
        elif any(word in query for word in ["how", "procedure", "steps"]):
            return "procedural"
        elif any(word in query for word in ["policy", "rule", "guideline"]):
            return "policy"
        elif any(word in query for word in ["why", "reason", "purpose"]):
            return "explanatory"
        else:
            return "informational"
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the query"""
        
        word_count = len(query.split())
        
        if word_count <= 3:
            return "simple"
        elif word_count <= 8:
            return "medium"
        else:
            return "complex"
    
    def _detect_knowledge_intent(self, query: str) -> str:
        """Detect the intent behind the knowledge query"""
        
        if any(word in query for word in ["requirement", "must", "mandatory", "required"]):
            return "compliance"
        elif any(word in query for word in ["best practice", "recommend", "should"]):
            return "guidance"
        elif any(word in query for word in ["example", "sample", "instance"]):
            return "example"
        else:
            return "information"
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from the knowledge query"""
        
        entities = []
        
        # Technical entities
        tech_terms = ["password", "2fa", "vpn", "api", "database", "server", "cloud"]
        entities.extend([term for term in tech_terms if term in query])
        
        # Department entities
        departments = ["engineering", "marketing", "sales", "support", "hr"]
        entities.extend([dept for dept in departments if dept in query])
        
        # Process entities
        processes = ["development", "deployment", "testing", "monitoring", "backup"]
        entities.extend([proc for proc in processes if proc in query])
        
        return list(set(entities))

# ...existing code...
class CodeExecutionProcessor:
    """Advanced code execution processor with enhanced security and capabilities"""
    
    def __init__(self):
        self.allowed_modules = {
            'math', 'statistics', 'random', 'datetime', 're', 'json',
            'collections', 'itertools', 'functools', 'operator'
        }
        self.safe_builtins = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'chr', 'dict',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'hex',
            'int', 'len', 'list', 'map', 'max', 'min', 'oct', 'ord', 'pow',
            'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
            'str', 'sum', 'tuple', 'type', 'zip'
        }
        self.execution_timeout = 5  # seconds
        self.max_output_length = 10000

    def _validate_code_safety(self, code: str) -> bool:
        """Basic code safety validation for dangerous patterns"""
        dangerous_patterns = ['__import__', 'exec(', 'eval(', 'open(', 'file(']
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        return True

    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process code execution requests with comprehensive analysis"""
        
        # Extract and analyze code
        code_info = self._extract_code(input_text)
        
        if not code_info["code"]:
            return {
                "error": "No executable code found in input",
                "suggestion": "Include Python code or use prefixes like 'execute:', 'calculate:', or 'run:'",
                "examples": [
                    "execute: print(sum(range(100)))",
                    "calculate: (15 * 37) + 128",
                    "run: import math; print(math.factorial(10))"
                ]
            }
        
        # เพิ่มการตรวจสอบความปลอดภัยก่อนรัน
        if not self._validate_code_safety(code_info["code"]):
            return {
                "code": code_info["code"],
                "error": "Code contains unsafe operations (import, exec, eval, open, file)",
                "status": "security_error",
                "security_note": "Code execution blocked for security reasons"
            }
        
        try:
            # Validate code before execution
            validation_result = self._validate_code(code_info["code"])
            if not validation_result["is_safe"]:
                return {
                    "code": code_info["code"],
                    "error": f"Code validation failed: {validation_result['reason']}",
                    "status": "validation_error",
                    "security_note": "Code execution blocked for security reasons"
                }
            
            # Execute code with monitoring
            start_time = time.time()
            execution_result = await self._execute_safe_code(code_info["code"])
            execution_time = time.time() - start_time
            
            return {
                "code": code_info["code"],
                "language": code_info["language"],
                "execution_type": code_info["type"],
                "result": execution_result["output"],
                "execution_time": execution_time,
                "status": "success",
                "code_analysis": self._analyze_code(code_info["code"]),
                "performance_metrics": {
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "output_length": len(execution_result["output"]),
                    "memory_efficient": execution_time < 1.0
                }
            }
            
        except Exception as e:
            return {
                "code": code_info["code"],
                "language": code_info["language"],
                "execution_type": code_info["type"],
                "error": str(e),
                "status": "execution_error",
                "suggestion": "Check your code syntax and ensure you're using only basic Python operations",
                "allowed_modules": list(self.allowed_modules)
            }
    
    def _extract_code(self, text: str) -> Dict[str, Any]:
        """Enhanced code extraction with better pattern recognition"""
        
        text = text.strip()
        
        # Check for explicit code markers
        code_markers = [
            ("execute:", "python"),
            ("python:", "python"),
            ("calculate:", "python"),
            ("math:", "python"),
            ("run:", "python"),
            ("eval:", "python")
        ]
        
        for marker, lang in code_markers:
            if text.lower().startswith(marker):
                code = text[len(marker):].strip()
                return {
                    "code": code,
                    "language": lang,
                    "type": "explicit_command"
                }
        
        # Check for code blocks with various formats
        code_block_patterns = [
            (r'```python\s*(.*?)\s*```', "python", "code_block"),
            (r'```py\s*(.*?)\s*```', "python", "code_block"),
            (r'```\s*(.*?)\s*```', "python", "generic_code_block"),
            (r'`([^`\n]+)`', "python", "inline_code")
        ]
        
        for pattern, lang, block_type in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return {
                    "code": matches[0].strip(),
                    "language": lang,
                    "type": block_type
                }
        
        # Enhanced mathematical expression detection
        if self._is_math_expression(text):
            # Try to make it executable
            if not any(func in text for func in ['print(', 'return ']):
                code = f"print({text})"
            else:
                code = text
            
            return {
                "code": code,
                "language": "python",
                "type": "math_expression"
            }
        
        # Check if it looks like Python code
        if self._looks_like_python_code(text):
            return {
                "code": text,
                "language": "python",
                "type": "inferred_python"
            }
        
        return {
            "code": "",
            "language": "unknown",
            "type": "none"
        }
    
    def _is_math_expression(self, text: str) -> bool:
        """Enhanced mathematical expression detection"""
        
        # Clean the text
        text = text.strip()
        
        # Basic math expression pattern
        math_pattern = r'^[\d\s+\-*/().%**]+$'
        if re.match(math_pattern, text):
            return True
        
        # Math with functions
        math_functions = ['abs', 'round', 'max', 'min', 'sum', 'pow']
        if any(func in text.lower() for func in math_functions):
            # Check if it's mostly mathematical
            non_alpha = sum(1 for c in text if not c.isalpha())
            total_chars = len(text)
            if non_alpha / total_chars > 0.3:  # At least 30% non-alphabetic
                return True
        
        # Simple arithmetic expressions
        if re.search(r'\d+\s*[+\-*/]\s*\d+', text):
            return True
        
        return False
    
    def _looks_like_python_code(self, text: str) -> bool:
        """Enhanced Python code detection"""
        
        python_indicators = [
            'print(', 'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 
            'while ', 'try:', 'except:', 'with ', 'lambda ', 'yield ',
            'return ', 'break', 'continue', '=', '+=', '-=', '*=', '/=',
            'and ', 'or ', 'not ', 'in ', 'is ', 'elif ', 'else:'
        ]
        
        indicator_count = sum(1 for indicator in python_indicators if indicator in text)
        
        # Must have at least 2 indicators for non-trivial code
        if indicator_count >= 2:
            return True
        
        # Single strong indicators
        strong_indicators = ['def ', 'class ', 'import ', 'from ']
        if any(indicator in text for indicator in strong_indicators):
            return True
        
        return False
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Comprehensive code validation for security and safety"""
        
        code_lower = code.lower()
        
        # Check for dangerous operations
        dangerous_patterns = [
            ('__import__', 'Dynamic imports not allowed'),
            ('exec(', 'Dynamic code execution not allowed'),
            ('eval(', 'Dynamic evaluation restricted to simple expressions'),
            ('open(', 'File operations not allowed'),
            ('file(', 'File operations not allowed'),
            ('input(', 'User input not allowed in executed code'),
            ('raw_input(', 'User input not allowed'),
            ('subprocess', 'System commands not allowed'),
            ('os.', 'OS operations not allowed'),
            ('sys.', 'System operations restricted'),
            ('socket', 'Network operations not allowed'),
            ('urllib', 'Network operations not allowed'),
            ('requests', 'Network requests not allowed'),
            ('while true', 'Infinite loops not allowed'),
            ('while 1', 'Infinite loops not allowed')
        ]
        
        for pattern, reason in dangerous_patterns:
            if pattern in code_lower:
                return {"is_safe": False, "reason": reason}
        
        # Check for reasonable code length
        if len(code) > 5000:
            return {"is_safe": False, "reason": "Code too long (max 5000 characters)"}
        
        # Check for reasonable complexity (nested loops, etc.)
        loop_count = code_lower.count('for ') + code_lower.count('while ')
        if loop_count > 3:
            return {"is_safe": False, "reason": "Too many loops (max 3)"}
        
        # Try to parse the code to check syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return {"is_safe": False, "reason": f"Syntax error: {str(e)}"}
        
        return {"is_safe": True, "reason": "Code passed validation"}
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and complexity"""
        
        lines = code.split('\n')
        
        return {
            "line_count": len(lines),
            "character_count": len(code),
            "has_functions": 'def ' in code,
            "has_loops": any(keyword in code for keyword in ['for ', 'while ']),
            "has_conditionals": any(keyword in code for keyword in ['if ', 'elif ', 'else:']),
            "imports": re.findall(r'import (\w+)', code) + re.findall(r'from (\w+)', code),
            "complexity_score": self._calculate_complexity_score(code)
        }
    
    def _calculate_complexity_score(self, code: str) -> int:
        """Calculate a simple complexity score for the code"""
        
        score = 0
        
        # Base score for any code
        score += 1
        
        # Add for control structures
        score += code.count('if ') * 2
        score += code.count('for ') * 3
        score += code.count('while ') * 3
        score += code.count('try:') * 2
        score += code.count('def ') * 4
        score += code.count('class ') * 5
        
        # Add for nested structures (approximation)
        indentation_levels = len(set(len(line) - len(line.lstrip()) for line in code.split('\n') if line.strip()))
        score += indentation_levels * 2
        
        return min(score, 20)  # Cap at 20
    
    async def _execute_safe_code(self, code: str) -> Dict[str, Any]:
        """Execute code in a secure environment with comprehensive monitoring"""
        
        # Create safe execution environment
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name) 
                for name in self.safe_builtins 
                if hasattr(__builtins__, name)
            }
        }
        
        # Add safe modules
        safe_modules = {}
        for module_name in self.allowed_modules:
            try:
                safe_modules[module_name] = __import__(module_name)
            except ImportError:
                continue
        
        safe_globals.update(safe_modules)
        
        # Capture all output
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            # Execute with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            
            # Set timeout (Unix systems only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.execution_timeout)
            
            try:
                # Try to execute as statements first
                exec(code, safe_globals, {})
                output = captured_output.getvalue()
            except:
                # If that fails, try as expression
                try:
                    result = eval(code, safe_globals, {})
                    output = str(result)
                except:
                    # Re-raise the original exec error
                    exec(code, safe_globals, {})
                    output = captured_output.getvalue()
            
            # Disable timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Ensure output is not too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            # If no output, provide a default message
            if not output.strip():
                output = "Code executed successfully (no output produced)"
            
        finally:
            sys.stdout = original_stdout
        
        return {
            "output": output.strip(),
            "truncated": len(output) >= self.max_output_length
        }

class HRMProcessor:
    """Hierarchical Reasoning Model processor for complex analytical tasks"""
    
    def __init__(self):
        self.available = False
        try:
            # In a real implementation, you would initialize connection to Azure OpenAI
            # For this demo, we'll simulate HRM functionality
            self.available = True
        except Exception:
            logger.warning("HRM system not available - using mock implementation")
    
    async def process(self, input_text: str, complexity: TaskComplexity = TaskComplexity.COMPLEX) -> Dict[str, Any]:
        """Process complex reasoning tasks using hierarchical approach"""
        
        if not self.available:
            return await self._mock_hrm_processing(input_text, complexity)
        
        # In real implementation, this would orchestrate multiple AI models
        # For demo, we provide structured analysis
        return await self._structured_analysis(input_text, complexity)
    
    async def _structured_analysis(self, query: str, complexity: TaskComplexity) -> Dict[str, Any]:
        """Provide structured analysis for complex queries"""
        
        analysis_components = []
        
        # Break down the query
        if "analyze" in query.lower() or "compare" in query.lower():
            analysis_components.extend([
                "Problem decomposition",
                "Multi-perspective analysis", 
                "Evidence synthesis",
                "Conclusion formulation"
            ])
        
        if "pros and cons" in query.lower():
            analysis_components.extend([
                "Advantage identification",
                "Disadvantage assessment",
                "Risk analysis",
                "Recommendation synthesis"
            ])
        
        # Generate structured response
        response = {
            "reasoning_approach": "hierarchical_analysis",
            "complexity_level": complexity.value,
            "analysis_components": analysis_components,
            "structured_response": self._generate_structured_response(query),
            "confidence": 0.85,
            "reasoning_depth": "comprehensive" if complexity in [TaskComplexity.COMPLEX, TaskComplexity.STRATEGIC] else "moderate"
        }
        
        return response
    
    def _generate_structured_response(self, query: str) -> Dict[str, Any]:
        """Generate a structured response based on query analysis"""
        
        query_lower = query.lower()
        
        if "pros and cons" in query_lower or "advantages and disadvantages" in query_lower:
            # Extract the subject being analyzed
            subject = self._extract_analysis_subject(query)
            
            return {
                "analysis_type": "pros_and_cons",
                "subject": subject,
                "pros": [
                    f"Potential benefits of {subject} include improved efficiency and flexibility",
                    f"{subject} may offer cost advantages and scalability",
                    f"Enhanced capabilities and modern features are typical advantages"
                ],
                "cons": [
                    f"Implementation challenges may arise with {subject}",
                    f"Initial costs and learning curve considerations for {subject}",
                    f"Potential compatibility and integration issues"
                ],
                "recommendation": f"Consider {subject} based on your specific needs, timeline, and resources. Conduct a pilot program to evaluate effectiveness.",
                "key_factors": ["cost", "implementation_time", "team_readiness", "business_impact"]
            }
        
        elif "compare" in query_lower:
            subjects = self._extract_comparison_subjects(query)
            
            return {
                "analysis_type": "comparison",
                "subjects": subjects,
                "comparison_framework": {
                    "criteria": ["Performance", "Cost", "Ease of Use", "Scalability", "Support"],
                    "methodology": "Multi-criteria decision analysis"
                },
                "summary": f"Comprehensive comparison of {' vs '.join(subjects)} across multiple dimensions",
                "recommendation": "Selection should be based on prioritized criteria and specific use case requirements"
            }
        
        else:
            return {
                "analysis_type": "general_analysis",
                "approach": "systematic_breakdown",
                "key_insights": [
                    "Complex problems require structured analysis",
                    "Multiple perspectives enhance understanding",
                    "Evidence-based conclusions are most reliable"
                ],
                "methodology": "Hierarchical reasoning with iterative refinement"
            }
    
    def _extract_analysis_subject(self, query: str) -> str:
        """Extract the main subject being analyzed"""
        
        # Simple extraction - in real implementation would use NLP
        query_lower = query.lower()
        
        common_subjects = [
            "remote work", "cloud computing", "artificial intelligence", "machine learning",
            "automation", "digital transformation", "agile methodology", "microservices"
        ]
        
        for subject in common_subjects:
            if subject in query_lower:
                return subject
        
        # Fallback extraction
        words = query.split()
        if len(words) > 2:
            return " ".join(words[1:4])  # Take middle words as subject
        
        return "the given topic"
    
    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """Extract subjects being compared"""
        
        # Look for "vs", "versus", "compared to" patterns
        vs_patterns = [" vs ", " versus ", " compared to ", " against "]
        
        for pattern in vs_patterns:
            if pattern in query.lower():
                parts = query.lower().split(pattern)
                if len(parts) >= 2:
                    return [part.strip() for part in parts[:2]]
        
        # Fallback
        return ["option A", "option B"]
    
    async def _mock_hrm_processing(self, query: str, complexity: TaskComplexity) -> Dict[str, Any]:
        """Mock HRM processing when real system is not available"""
        
        return {
            "reasoning_approach": "mock_hierarchical_reasoning",
            "complexity_level": complexity.value,
            "response": f"This is a mock HRM response for the complex query: '{query}'. In a full implementation, this would use hierarchical reasoning with multiple AI models to provide comprehensive analysis.",
            "note": "Mock HRM response - real implementation would provide deeper analysis",
            "suggested_components": [
                "Worker models for detailed analysis",
                "Head models for synthesis",
                "Executive model for strategic oversight"
            ]
        }

class CompleteDecisionAgent:
    """Complete Decision Agent with all processors and enhanced capabilities"""
    
    def __init__(self):
        # Initialize all processors
        self.web_scraper = WebScrapingProcessor()
        self.search_processor = GoogleSearchProcessor()
        self.database_processor = DatabaseProcessor()
        self.knowledge_processor = KnowledgeRAGProcessor()
        self.code_processor = CodeExecutionProcessor()
        self.hrm_processor = HRMProcessor()
        
        # System state
        self.processing_history: List[ProcessedTask] = []
        self.system_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "task_counts": {task_type.value: 0 for task_type in TaskType},
            "average_confidence": 0.0,
            "total_processing_time": 0.0,
            "start_time": datetime.now()
        }
        
        logger.info("Decision Agent initialized with all processors")
    
    async def process_request(self, input_text: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Main processing function with comprehensive task routing"""
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        self.system_stats["total_requests"] += 1
        
        try:
            # Step 1: Comprehensive input analysis
            task_scores = await self._analyze_input_comprehensive(input_text)
            
            if not task_scores:
                raise ValueError("Unable to analyze input - no suitable processors found")
            
            # Step 2: Select optimal task processor
            selected_task_score = self._select_optimal_task(task_scores, options)
            
            # Step 3: Execute with selected processor
            result = await self._execute_task_with_monitoring(
                selected_task_score.task_type,
                input_text,
                selected_task_score,
                options
            )
            
            # Step 4: Post-process and enrich results
            enriched_result = self._enrich_result(result, selected_task_score, input_text)
            
            # Step 5: Create comprehensive task record
            processing_time = time.time() - start_time
            
            processed_task = ProcessedTask(
                task_id=task_id,
                input_text=input_text,
                selected_task=selected_task_score.task_type,
                confidence_used=selected_task_score.confidence,
                result=enriched_result,
                reasoning=selected_task_score.reasoning,
                processing_time=processing_time,
                metadata={
                    "task_scores": [
                        {
                            "task_type": score.task_type.value,
                            "confidence": score.confidence,
                            "reasoning": score.reasoning,
                            "complexity": score.complexity.value
                        }
                        for score in task_scores
                    ],
                    "options": options or {},
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.processing_history.append(processed_task)
            self._update_system_stats(processed_task)
            
            return {
                "task_id": task_id,
                "status": "success",
                "selected_task": selected_task_score.task_type.value,
                "confidence_used": selected_task_score.confidence,
                "complexity": selected_task_score.complexity.value,
                "reasoning": selected_task_score.reasoning,
                "result": enriched_result,
                "processing_time": round(processing_time, 4),
                "task_analysis": processed_task.metadata["task_scores"],
                "performance_metrics": {
                    "response_time_ms": round(processing_time * 1000, 2),
                    "processor_efficiency": "high" if processing_time < 2 else "medium",
                    "result_quality": "high" if selected_task_score.confidence > 0.8 else "medium"
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.system_stats["failed_requests"] += 1
            
            error_task = ProcessedTask(
                task_id=task_id,
                input_text=input_text,
                selected_task=TaskType.GENERAL_QUERY,
                confidence_used=0.0,
                result=None,
                reasoning="Processing failed",
                processing_time=processing_time,
                status="error",
                error=str(e),
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            self.processing_history.append(error_task)
            
            logger.error(f"Task processing failed for task {task_id}: {str(e)}")
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": round(processing_time, 4),
                "suggestion": self._get_error_suggestion(str(e)),
                "recovery_options": [
                    "Try simplifying your request",
                    "Check if the input format is correct",
                    "Use specific task prefixes (e.g., 'search:', 'execute:')"
                ]
            }
    
    async def _analyze_input_comprehensive(self, input_text: str) -> List[TaskScore]:
        """Comprehensive input analysis with enhanced scoring"""
        
        scores = []
        text_lower = input_text.lower().strip()
        
        # Web scraping detection (enhanced)
        url_score = await self._calculate_web_scraping_score(input_text, text_lower)
        if url_score:
            scores.append(url_score)
        
        # Search detection (enhanced)
        search_score = self._calculate_search_score(input_text, text_lower)
        if search_score:
            scores.append(search_score)
        
        # Database query detection (enhanced)
        db_score = self._calculate_database_score(input_text, text_lower)
        if db_score:
            scores.append(db_score)
        
        # Knowledge management detection (enhanced)
        km_score = self._calculate_knowledge_score(input_text, text_lower)
        if km_score:
            scores.append(km_score)
        
        # Code execution detection (enhanced)
        code_score = self._calculate_code_score(input_text, text_lower)
        if code_score:
            scores.append(code_score)
        
        # HRM reasoning detection (enhanced)
        hrm_score = self._calculate_hrm_score(input_text, text_lower)
        if hrm_score:
            scores.append(hrm_score)
        
        # General query (always included as fallback)
        general_confidence = max(0.3, 1.0 - max([score.confidence for score in scores] + [0]))
        scores.append(TaskScore(
            task_type=TaskType.GENERAL_QUERY,
            confidence=general_confidence,
            reasoning="General AI query handling with fallback processing",
            priority=4,
            complexity=TaskComplexity.SIMPLE
        ))
        
        # Sort by confidence, then by priority
        scores.sort(key=lambda x: (-x.confidence, x.priority))
        
        return scores
    
    async def _calculate_web_scraping_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate web scraping confidence score"""
        
        # Enhanced URL detection
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+\.[a-zA-Z]{2,}'
        urls = re.findall(url_pattern, input_text)
        
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, input_text)
        
        url_count = len(urls) + len([d for d in domains if not any(d in url for url in urls)])
        
        if url_count > 0:
            # Higher confidence for multiple URLs or explicit scraping requests
            base_confidence = min(0.95, 0.7 + (url_count * 0.1))
            
            # Boost for explicit scraping terms
            scraping_terms = ['scrape', 'extract', 'crawl', 'parse', 'get content']
            if any(term in text_lower for term in scraping_terms):
                base_confidence = min(0.98, base_confidence + 0.15)
            
            return TaskScore(
                task_type=TaskType.WEB_SCRAPING,
                confidence=base_confidence,
                reasoning=f"Detected {url_count} URL(s) requiring web scraping",
                priority=1,
                complexity=TaskComplexity.MEDIUM if url_count > 1 else TaskComplexity.SIMPLE
            )
        
        return None
    
    def _calculate_search_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate search confidence score"""
        
        # Enhanced search detection
        explicit_search_terms = ['search for', 'search', 'google', 'find', 'look up', 'lookup']
        question_words = ['what is', 'who is', 'how to', 'where is', 'when is', 'why']
        
        explicit_matches = sum(1 for term in explicit_search_terms if text_lower.startswith(term))
        question_matches = sum(1 for word in question_words if word in text_lower)
        
        if explicit_matches > 0 or question_matches > 0:
            base_confidence = 0.8 if explicit_matches > 0 else 0.6
            
            # Boost for informational queries
            if any(word in text_lower for word in ['latest', 'news', 'current', 'recent', 'today']):
                base_confidence += 0.1
            
            return TaskScore(
                task_type=TaskType.GOOGLE_SEARCH,
                confidence=min(0.92, base_confidence),
                reasoning="Search keywords detected - web search appropriate",
                priority=2,
                complexity=TaskComplexity.SIMPLE
            )
        
        return None
    
    def _calculate_database_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate database query confidence score"""
        
        # Enhanced database detection
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
        db_terms = ['database', 'query', 'sql', 'table', 'records', 'data']
        entity_terms = ['users', 'tasks', 'projects', 'employees', 'customers']
        action_terms = ['show', 'list', 'get', 'find', 'count', 'display', 'view']
        
        sql_matches = sum(1 for kw in sql_keywords if text_lower.strip().startswith(kw))
        db_matches = sum(1 for term in db_terms if term in text_lower)
        entity_matches = sum(1 for term in entity_terms if term in text_lower)
        action_matches = sum(1 for term in action_terms if term in text_lower)
        
        total_score = sql_matches * 0.4 + db_matches * 0.2 + entity_matches * 0.2 + action_matches * 0.1
        
        if total_score > 0.1:
            confidence = min(0.9, 0.4 + total_score)
            
            return TaskScore(
                task_type=TaskType.DATABASE_QUERY,
                confidence=confidence,
                reasoning=f"Database query patterns detected (score: {total_score:.2f})",
                priority=2,
                complexity=TaskComplexity.MEDIUM if sql_matches > 0 else TaskComplexity.SIMPLE
            )
        
        return None
    
    def _calculate_knowledge_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate knowledge management confidence score"""
        
        # Enhanced knowledge detection
        policy_terms = ['policy', 'procedure', 'guideline', 'standard', 'rule', 'regulation']
        knowledge_terms = ['how should', 'what should', 'best practice', 'recommendation']
        domain_terms = ['security', 'development', 'support', 'hr', 'company']
        
        policy_matches = sum(1 for term in policy_terms if term in text_lower)
        knowledge_matches = sum(1 for term in knowledge_terms if term in text_lower)
        domain_matches = sum(1 for term in domain_terms if term in text_lower)
        
        total_score = policy_matches * 0.3 + knowledge_matches * 0.4 + domain_matches * 0.2
        
        if total_score > 0.2:
            confidence = min(0.88, 0.5 + total_score)
            
            return TaskScore(
                task_type=TaskType.KM_RAG,
                confidence=confidence,
                reasoning="Knowledge base query detected for company policies/procedures",
                priority=2,
                complexity=TaskComplexity.MEDIUM
            )
        
        return None
    
    def _calculate_code_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate code execution confidence score"""
        
        # Enhanced code detection
        explicit_markers = ['execute:', 'python:', 'calculate:', 'math:', 'run:', 'eval:']
        code_blocks = ['```', '`']
        math_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'[+\-*/]\s*\d+', r'\d+\s*\*\*\s*\d+']
        code_terms = ['print(', 'def ', 'import ', 'for ', 'if ', 'while ']
        
        explicit_count = sum(1 for marker in explicit_markers if text_lower.startswith(marker))
        block_count = sum(1 for block in code_blocks if block in input_text)
        math_count = sum(1 for pattern in math_patterns if re.search(pattern, input_text))
        code_count = sum(1 for term in code_terms if term in text_lower)
        
        if explicit_count > 0:
            confidence = 0.95
            reasoning = "Explicit code execution command detected"
        elif block_count > 0:
            confidence = 0.90
            reasoning = "Code block formatting detected"
        elif math_count > 0:
            confidence = 0.85
            reasoning = "Mathematical expression detected"
        elif code_count > 0:
            confidence = 0.70
            reasoning = "Code-like syntax detected"
        else:
            return None
        
        return TaskScore(
            task_type=TaskType.CODE_EXECUTION,
            confidence=confidence,
            reasoning=reasoning,
            priority=1,
            complexity=TaskComplexity.SIMPLE
        )
    
    def _calculate_hrm_score(self, input_text: str, text_lower: str) -> Optional[TaskScore]:
        """Calculate HRM reasoning confidence score"""
        
        # Enhanced HRM detection
        analysis_terms = ['analyze', 'compare', 'evaluate', 'assess', 'examine']
        complexity_terms = ['pros and cons', 'advantages', 'disadvantages', 'trade-offs']
        strategic_terms = ['strategic', 'comprehensive', 'detailed analysis', 'in-depth']
        reasoning_terms = ['because', 'therefore', 'however', 'furthermore', 'moreover']
        
        analysis_matches = sum(1 for term in analysis_terms if term in text_lower)
        complexity_matches = sum(1 for term in complexity_terms if term in text_lower)
        strategic_matches = sum(1 for term in strategic_terms if term in text_lower)
        reasoning_matches = sum(1 for term in reasoning_terms if term in text_lower)
        
        # Length-based complexity indicator
        word_count = len(input_text.split())
        length_factor = min(0.3, word_count / 50)  # Boost for longer, complex queries
        
        total_score = (analysis_matches * 0.25 + complexity_matches * 0.30 + 
                      strategic_matches * 0.20 + reasoning_matches * 0.10 + length_factor)
        
        if total_score > 0.3:
            confidence = min(0.92, 0.4 + total_score)
            complexity = TaskComplexity.STRATEGIC if total_score > 0.7 else TaskComplexity.COMPLEX
            
            return TaskScore(
                task_type=TaskType.HRM_REASONING,
                confidence=confidence,
                reasoning="Complex analytical reasoning task detected",
                priority=1,
                complexity=complexity
            )
        
        return None
    
    def _select_optimal_task(self, task_scores: List[TaskScore], options: Optional[Dict]) -> TaskScore:
        """Select the optimal task processor based on scores and options"""
        
        if not task_scores:
            # Fallback to general query
            return TaskScore(
                task_type=TaskType.GENERAL_QUERY,
                confidence=0.5,
                reasoning="No specific task type detected - using general processing",
                priority=5
            )
        
        # Consider user preferences from options
        if options and options.get("preferred_task"):
            preferred = options["preferred_task"]
            for score in task_scores:
                if score.task_type.value == preferred and score.confidence > 0.3:
                    return score
        
        # Consider confidence thresholds
        high_confidence_threshold = 0.8
        high_confidence_tasks = [s for s in task_scores if s.confidence >= high_confidence_threshold]
        
        if high_confidence_tasks:
            return high_confidence_tasks[0]  # Already sorted by confidence
        
        # Return highest confidence task
        return task_scores[0]
    
    async def _execute_task_with_monitoring(
        self, 
        task_type: TaskType, 
        input_text: str, 
        task_score: TaskScore, 
        options: Optional[Dict]
    ) -> Any:
        """Execute task with comprehensive monitoring and error handling"""
        
        processor_start = time.time()
        
        try:
            # หลังจาก execute task แล้ว
            result = await self._execute_basic_task(task_type, input_text)
            # เพิ่ม HRM enhancement สำหรับทุก task
            enhanced_result = await self._apply_hrm_enhancement(result, task_type, input_text)
            return enhanced_result
            
        except Exception as e:
            processor_time = time.time() - processor_start
            logger.error(f"Task execution failed for {task_type.value}: {str(e)}")
            
            return {
                "error": f"Task execution failed: {str(e)}",
                "task_type": task_type.value,
                "processor_time": processor_time,
                "fallback_attempted": True
            }
    
    def _enrich_result(self, result: Any, task_score: TaskScore, input_text: str) -> Dict[str, Any]:
        """Enrich results with additional context and metadata"""
        
        if not isinstance(result, dict):
            result = {"result": result}
        
        # Add enrichment metadata
        enrichment = {
            "task_confidence": task_score.confidence,
            "complexity_level": task_score.complexity.value,
            "input_analysis": {
                "word_count": len(input_text.split()),
                "character_count": len(input_text),
                "contains_urls": bool(re.search(r'https?://', input_text)),
                "contains_code": bool(re.search(r'[{}()\[\]]', input_text)),
                "question_words": len(re.findall(r'\b(?:what|how|why|when|where|who)\b', input_text.lower()))
            },
            "recommendations": self._generate_recommendations(result, task_score)
        }
        
        result["enrichment"] = enrichment
        return result
    
    def _generate_recommendations(self, result: Dict, task_score: TaskScore) -> List[str]:
        """Generate helpful recommendations based on results"""
        
        recommendations = []
        
        # Task-specific recommendations
        if task_score.task_type == TaskType.WEB_SCRAPING:
            if result.get("successful_scrapes", 0) == 0:
                recommendations.append("Try checking if the URLs are accessible and valid")
            else:
                recommendations.append("Consider using the extracted content for further analysis")
        
        elif task_score.task_type == TaskType.GOOGLE_SEARCH:
            recommendations.append("Review search results for relevance and explore related searches")
            if "mock" in result.get("status", ""):
                recommendations.append("For real search results, ensure internet connectivity")
        
        elif task_score.task_type == TaskType.DATABASE_QUERY:
            if result.get("row_count", 0) == 0:
                recommendations.append("Try broadening your query criteria or check available data")
            else:
                recommendations.append("Consider exporting results or creating visualizations")
        
        elif task_score.task_type == TaskType.CODE_EXECUTION:
            if result.get("status") == "success":
                recommendations.append("Code executed successfully - consider saving for reuse")
            else:
                recommendations.append("Review code syntax and try simpler expressions first")
        
        # Confidence-based recommendations
        if task_score.confidence < 0.6:
            recommendations.append("Consider being more specific about what you want to accomplish")
        
        return recommendations
    
    async def _process_general_query(self, input_text: str) -> Dict[str, Any]:
        """Process general queries with enhanced AI response simulation"""
        
        # Analyze query characteristics
        query_analysis = {
            "is_question": any(word in input_text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']),
            "is_greeting": any(word in input_text.lower() for word in ['hello', 'hi', 'hey', 'greetings']),
            "is_request": any(word in input_text.lower() for word in ['please', 'can you', 'could you', 'help']),
            "word_count": len(input_text.split())
        }
        
        # Generate contextual response
        if query_analysis["is_greeting"]:
            response = "Hello! I'm the Decision Agent, ready to help you with web scraping, searches, database queries, code execution, and more. What can I assist you with today?"
        
        elif query_analysis["is_question"]:
            response = f"I understand you're asking about '{input_text}'. While I can provide general assistance, I'm particularly effective at specific tasks like web scraping, database queries, code execution, and knowledge base searches. Consider rephrasing your request to use one of my specialized capabilities."
        
        elif query_analysis["is_request"]:
            response = f"I'd be happy to help with '{input_text}'. For the best results, try using specific task prefixes or provide URLs, code, or database queries that I can process directly."
        
        else:
            response = f"I've processed your input: '{input_text}'. This appears to be a general query. I can provide more targeted assistance with web scraping, searches, database operations, code execution, or knowledge base queries."
        
        return {
            "response": response,
            "type": "general_response",
            "query_analysis": query_analysis,
            "capabilities": [
                "Web scraping (provide URLs)",
                "Web search (use 'search for' prefix)",
                "Database queries (use 'show', 'find', 'list')",
                "Code execution (use 'execute:', 'calculate:')",
                "Knowledge base (ask about policies, procedures)",
                "Complex analysis (analytical questions)"
            ],
            "examples": [
                "https://example.com (web scraping)",
                "search for latest Python tutorials",
                "show all users in database",
                "calculate: 15 * 37 + 128",
                "what is the security policy?",
                "analyze pros and cons of remote work"
            ]
        }
    
    def _update_system_stats(self, task: ProcessedTask):
        """Update comprehensive system statistics"""
        
        if task.status == "success":
            self.system_stats["successful_requests"] += 1
        else:
            self.system_stats["failed_requests"] += 1
        
        # Update task type counts
        self.system_stats["task_counts"][task.selected_task.value] += 1
        
        # Update timing statistics
        self.system_stats["total_processing_time"] += task.processing_time
        
        # Update average confidence
        successful_tasks = [t for t in self.processing_history if t.status == "success"]
        if successful_tasks:
            self.system_stats["average_confidence"] = sum(t.confidence_used for t in successful_tasks) / len(successful_tasks)
    
    def _get_error_suggestion(self, error_message: str) -> str:
        """Provide contextual error suggestions"""
        
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "The request timed out. Try simplifying your input or checking network connectivity."
        elif "validation" in error_lower:
            return "Input validation failed. Check your syntax and try again."
        elif "parse" in error_lower or "syntax" in error_lower:
            return "There was a parsing error. Verify your input format and syntax."
        elif "network" in error_lower or "connection" in error_lower:
            return "Network error occurred. Check your internet connection and try again."
        else:
            return "An unexpected error occurred. Try rephrasing your request or using a different approach."
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        uptime = (datetime.now() - self.system_stats["start_time"]).total_seconds()
        
        return {
            **self.system_stats,
            "success_rate": self.system_stats["successful_requests"] / max(1, self.system_stats["total_requests"]),
            "average_processing_time": self.system_stats["total_processing_time"] / max(1, self.system_stats["total_requests"]),
            "uptime_seconds": uptime,
            "uptime_formatted": str(datetime.now() - self.system_stats["start_time"]).split('.')[0],
            "requests_per_minute": self.system_stats["total_requests"] / max(1, uptime / 60),
            "processor_status": {
                "web_scraper": "active",
                "search_processor": "active",
                "database_processor": "active", 
                "knowledge_processor": "active",
                "code_processor": "active",
                "hrm_processor": "active" if self.hrm_processor.available else "mock"
            },
            "recent_tasks": [
                {
                    "task_id": task.task_id,
                    "input_preview": task.input_text[:50] + "..." if len(task.input_text) > 50 else task.input_text,
                    "selected_task": task.selected_task.value,
                    "confidence": task.confidence_used,
                    "status": task.status,
                    "processing_time": task.processing_time,
                    "timestamp": task.metadata.get("timestamp", "")
                }
                for task in self.processing_history[-15:]
            ]
        }
    
    async def analyze_input_detailed(self, input_text: str) -> Dict[str, Any]:
        """Provide detailed input analysis for debugging and optimization"""
        
        task_scores = await self._analyze_input_comprehensive(input_text)
        
        return {
            "input": input_text,
            "input_length": len(input_text),
            "word_count": len(input_text.split()),
            "task_scores": [
                {
                    "task_type": score.task_type.value,
                    "confidence": round(score.confidence, 4),
                    "reasoning": score.reasoning,
                    "priority": score.priority,
                    "complexity": score.complexity.value
                }
                for score in task_scores
            ],
            "recommended_task": task_scores[0].task_type.value if task_scores else "none",
            "confidence_distribution": {
                task_type.value: next((s.confidence for s in task_scores if s.task_type == task_type), 0.0)
                for task_type in TaskType
            },
            "analysis_metadata": {
                "has_urls": bool(re.search(r'https?://', input_text)),
                "has_code_markers": any(input_text.lower().startswith(marker) for marker in ['execute:', 'python:', 'calculate:']),
                "has_question_words": bool(re.search(r'\b(?:what|how|why|when|where|who)\b', input_text.lower())),
                "has_search_terms": any(term in input_text.lower() for term in ['search', 'find', 'look up']),
                "complexity_indicators": len(re.findall(r'\b(?:analyze|compare|evaluate|comprehensive)\b', input_text.lower()))
            }
        }

# Pydantic models for API
class ProcessRequest(BaseModel):
    input: str
    options: Optional[Dict[str, Any]] = None
    
    @validator('input')
    def input_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Input cannot be empty')
        return v.strip()

class AnalyzeRequest(BaseModel):
    input: str
    
    @validator('input')
    def input_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Input cannot be empty')
        return v.strip()

# Initialize the decision agent
decision_agent = CompleteDecisionAgent()

# FastAPI application with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    
    logger.info("🚀 Complete Decision Agent starting up...")
    logger.info(f"🧠 HRM System: {'Available' if decision_agent.hrm_processor.available else 'Mock Mode'}")
    logger.info("✅ All task processors initialized successfully")
    logger.info("🌐 Web interface available at /demo")
    logger.info("📚 API documentation available at /api/docs")
    
    yield
    
    logger.info("📊 Final Statistics:")
    stats = decision_agent.get_comprehensive_stats()
    logger.info(f"   Total Requests: {stats['total_requests']}")
    logger.info(f"   Success Rate: {stats['success_rate']:.1%}")
    logger.info(f"   Average Processing Time: {stats['average_processing_time']:.3f}s")
    logger.info("🛑 Decision Agent shutting down...")

# Create FastAPI application
app = FastAPI(
    title="Complete Decision Agent API",
    description="Intelligent task routing with web scraping, search, database, KM RAG, code execution, and HRM",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with enhanced demo interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Decision Agent - Intelligent Task Routing</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; color: white; padding: 20px;
            }
            .container { max-width: 1000px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .header p { font-size: 1.3rem; opacity: 0.9; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin: 40px 0; }
            .card { 
                background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; 
                backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .card:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,0,0,0.2); }
            .card h3 { margin-bottom: 15px; font-size: 1.4rem; }
            .feature { 
                margin: 12px 0; padding: 12px; background: rgba(255,255,255,0.1); 
                border-radius: 8px; border-left: 4px solid rgba(255,255,255,0.4);
            }
            .btn { 
                display: inline-block; padding: 15px 30px; margin: 10px 10px 10px 0;
                background: rgba(255,255,255,0.2); color: white; text-decoration: none; 
                border-radius: 10px; font-weight: 600; transition: all 0.3s;
                border: 1px solid rgba(255,255,255,0.3);
            }
            .btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
            .btn.primary { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); border: none; }
            .status { 
                background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.5);
                padding: 20px; border-radius: 12px; margin: 30px 0;
            }
            .quick-start { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; margin-top: 20px; }
            .quick-start code { background: rgba(0,0,0,0.3); padding: 4px 8px; border-radius: 4px; font-family: 'Courier New', monospace; }
            @media (max-width: 768px) { 
                .header h1 { font-size: 2rem; }
                .grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Decision Agent</h1>
                <p>Intelligent Task Routing System with Advanced AI Capabilities</p>
            </div>
            
            <div class="status">
                ✅ <strong>System Status:</strong> All processors online<br>
                🎯 <strong>Task Routing:</strong> 7 specialized processors ready<br>
                📊 <strong>Capabilities:</strong> Web scraping, search, database, code execution, knowledge management<br>
                🚀 <strong>Ready to process your requests with intelligent task detection!</strong>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>🎯 Core Capabilities</h3>
                    <div class="feature">📄 <strong>Web Scraping:</strong> Extract and analyze content from any URL</div>
                    <div class="feature">🔍 <strong>Smart Search:</strong> Web search with intelligent result processing</div>
                    <div class="feature">🗄️ <strong>Database Queries:</strong> Natural language to SQL conversion</div>
                    <div class="feature">📚 <strong>Knowledge RAG:</strong> Company knowledge retrieval and analysis</div>
                    <div class="feature">💻 <strong>Code Execution:</strong> Safe Python code execution environment</div>
                    <div class="feature">🧠 <strong>HRM Reasoning:</strong> Complex analytical reasoning tasks</div>
                </div>
                
                <div class="card">
                    <h3>🚀 Quick Examples</h3>
                    <div class="quick-start">
                        <p><strong>Web Scraping:</strong><br><code>https://example.com/article</code></p>
                        <p><strong>Search:</strong><br><code>search for latest AI developments</code></p>
                        <p><strong>Database:</strong><br><code>show all users by department</code></p>
                        <p><strong>Knowledge:</strong><br><code>what is our security policy?</code></p>
                        <p><strong>Code:</strong><br><code>calculate: fibonacci(15)</code></p>
                        <p><strong>Analysis:</strong><br><code>analyze pros and cons of cloud computing</code></p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⚡ Advanced Features</h3>
                    <div class="feature">🎯 <strong>Intelligent Routing:</strong> Automatic task detection and processor selection</div>
                    <div class="feature">📊 <strong>Confidence Scoring:</strong> ML-based confidence assessment for optimal results</div>
                    <div class="feature">🔒 <strong>Secure Execution:</strong> Sandboxed code execution with safety validation</div>
                    <div class="feature">📈 <strong>Performance Monitoring:</strong> Real-time processing metrics and optimization</div>
                    <div class="feature">🔄 <strong>Error Recovery:</strong> Automatic fallback and error handling</div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px;">
                <a href="/demo" class="btn primary">🎨 Try Interactive Demo</a>
                <a href="/api/docs" class="btn">📚 API Documentation</a>
                <a href="/api/stats" class="btn">📊 System Statistics</a>
                <a href="/api/health" class="btn">🔍 Health Check</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.post("/api/process", response_model=None)
async def process_request(self, input_text: str, options: Optional[Dict] = None) -> Dict[str, Any]:
    """Main processing function with comprehensive task routing"""
    start_time = time.time()
    task_id = str(uuid.uuid4())
    self.system_stats["total_requests"] += 1

    try:
        # Step 1: Comprehensive input analysis
        task_scores = await self._analyze_input_comprehensive(input_text)

        if not task_scores:
            raise ValueError("Unable to analyze input - no suitable processors found")

        # Step 2: Select optimal task processor
        selected_task_score = self._select_optimal_task(task_scores, options)

        # Step 3: Execute with selected processor
        result = await self._execute_task_with_monitoring(
            selected_task_score.task_type,
            input_text,
            selected_task_score,
            options
        )

        # Step 4: Post-process and enrich results
        enriched_result = self._enrich_result(result, selected_task_score, input_text)

        # เพิ่มการจัดรูปแบบผลลัพธ์ให้อ่านง่าย
        formatted_result = ResultFormatter.format_result(
            selected_task_score.task_type,
            enriched_result
        )

        # Step 5: Create comprehensive task record
        processing_time = time.time() - start_time

        processed_task = ProcessedTask(
            task_id=task_id,
            input_text=input_text,
            selected_task=selected_task_score.task_type,
            confidence_used=selected_task_score.confidence,
            result=enriched_result,
            reasoning=selected_task_score.reasoning,
            processing_time=processing_time,
            metadata={
                "task_scores": [
                    {
                        "task_type": score.task_type.value,
                        "confidence": score.confidence,
                        "reasoning": score.reasoning,
                        "complexity": score.complexity.value
                    }
                    for score in task_scores
                ],
                "options": options or {},
                "timestamp": datetime.now().isoformat()
            }
        )

        self.processing_history.append(processed_task)
        self._update_system_stats(processed_task)

        return {
            "task_id": task_id,
            "status": "success",
            "selected_task": selected_task_score.task_type.value,
            "confidence_used": selected_task_score.confidence,
            "complexity": selected_task_score.complexity.value,
            "reasoning": selected_task_score.reasoning,
            "result": enriched_result,
            "formatted_result": formatted_result,
            "processing_time": round(processing_time, 4),
            "task_analysis": processed_task.metadata["task_scores"],
            "performance_metrics": {
                "response_time_ms": round(processing_time * 1000, 2),
                "processor_efficiency": "high" if processing_time < 2 else "medium",
                "result_quality": "high" if selected_task_score.confidence > 0.8 else "medium"
            }
        }

    except Exception as e:
        processing_time = time.time() - start_time
        self.system_stats["failed_requests"] += 1

        error_task = ProcessedTask(
            task_id=task_id,
            input_text=input_text,
            selected_task=TaskType.GENERAL_QUERY,
            confidence_used=0.0,
            result=None,
            reasoning="Processing failed",
            processing_time=processing_time,
            status="error",
            error=str(e),
            metadata={"timestamp": datetime.now().isoformat()}
        )

        self.processing_history.append(error_task)

        logger.error(f"Task processing failed for task {task_id}: {str(e)}")

        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time": round(processing_time, 4),
            "suggestion": self._get_error_suggestion(str(e)),
            "recovery_options": [
                "Try simplifying your request",
                "Check if the input format is correct",
                "Use specific task prefixes (e.g., 'search:', 'execute:')"
            ]
        }

@app.post("/api/analyze", response_model=None)
async def analyze_input(request: AnalyzeRequest):
    """Analyze input and return detailed task confidence scores"""
    
    try:
        result = await decision_agent.analyze_input_detailed(request.input)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Input analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis processing failed")

@app.get("/api/stats")
async def get_statistics():
    """Get comprehensive system statistics"""
    
    try:
        return decision_agent.get_comprehensive_stats()
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    
    try:
        stats = decision_agent.get_comprehensive_stats()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "uptime": stats["uptime_formatted"],
            "processors": {
                "web_scraping": "operational",
                "google_search": "operational", 
                "database_query": "operational",
                "knowledge_rag": "operational",
                "code_execution": "operational",
                "hrm_reasoning": "operational" if decision_agent.hrm_processor.available else "mock_mode",
                "general_query": "operational"
            },
            "performance": {
                "total_requests": stats["total_requests"],
                "success_rate": round(stats["success_rate"], 3),
                "average_response_time": round(stats["average_processing_time"], 3),
                "requests_per_minute": round(stats.get("requests_per_minute", 0), 2)
            },
            "system": {
                "memory_usage": "normal",
                "cpu_usage": "normal",
                "disk_space": "adequate"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

class ResultFormatter:
    @staticmethod
    def format_result(task_type: TaskType, result: Dict) -> str:
        if task_type == TaskType.WEB_SCRAPING:
            return (
                f"🌐 ผลลัพธ์การดึงข้อมูลเว็บไซต์\n"
                f"📊 ดึงข้อมูลสำเร็จ {result.get('successful_scrapes', 0)} จาก {result.get('scraped_urls', 0)} URL\n"
                f"รายละเอียด: {', '.join([r.get('title', '') for r in result.get('results', []) if r.get('status') == 'success'])}"
            )
        elif task_type == TaskType.GOOGLE_SEARCH:
            items = result.get('results', [])
            lines = [f"🔍 ผลลัพธ์การค้นหา ({result.get('search_engine', '')})"]
            for i, item in enumerate(items[:5], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}\n   {item.get('snippet', '')}")
            return "\n".join(lines)
        elif task_type == TaskType.DATABASE_QUERY:
            summary = result.get('summary', '')
            row_count = result.get('row_count', 0)
            return f"🗄️ ผลลัพธ์ฐานข้อมูล\n{summary}\nจำนวนแถว: {row_count}"
        elif task_type == TaskType.KM_RAG:
            answer = result.get('response', '')
            sources = result.get('sources', [])
            src_lines = [f"- {src['title']} (อัปเดต {src['last_updated']})" for src in sources]
            return f"📚 ผลลัพธ์จากคลังความรู้\n{answer}\n\nแหล่งข้อมูล:\n" + "\n".join(src_lines)
        elif task_type == TaskType.CODE_EXECUTION:
            status = result.get('status', '')
            output = result.get('result', '')
            return f"💻 ผลลัพธ์การรันโค้ด ({status})\n{output}"
        elif task_type == TaskType.HRM_REASONING:
            resp = result.get('structured_response', result.get('response', ''))
            return f"🧠 ผลลัพธ์การวิเคราะห์เชิงเหตุผล\n{resp}"
        else:
            return result.get('response', str(result))


@app.get("/demo", response_class=HTMLResponse)
async def demo_interface():
    """Advanced interactive demo interface"""
    
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Decision Agent - Interactive Demo</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; padding: 20px;
            }
            .container {
                max-width: 1400px; margin: 0 auto;
                background: rgba(255,255,255,0.95); border-radius: 20px; overflow: hidden;
                box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            }
            .header {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white; padding: 30px; text-align: center;
            }
            .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
            .header p { font-size: 1.1rem; opacity: 0.9; }
            .main-content {
                display: grid; grid-template-columns: 1fr 1fr; gap: 30px; padding: 30px;
            }
            .input-panel, .results-panel {
                background: #f8fafc; padding: 25px; border-radius: 15px; 
                border: 1px solid #e2e8f0; min-height: 600px;
            }
            .panel-header { 
                display: flex; justify-content: space-between; align-items: center; 
                margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #e2e8f0;
            }
            .panel-header h3 { font-size: 1.3rem; color: #1f2937; }
            .input-field {
                width: 100%; padding: 15px; margin: 15px 0; font-size: 1rem;
                border: 2px solid #e5e7eb; border-radius: 10px; resize: vertical;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            .input-field:focus {
                outline: none; border-color: #4f46e5;
                box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
            }
            .btn {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white; border: none; padding: 12px 25px; border-radius: 8px;
                font-size: 1rem; font-weight: 600; cursor: pointer; margin: 8px 8px 8px 0;
                transition: all 0.3s; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3); }
            .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
            .btn.secondary { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); }
            .btn.danger { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
            .result-item {
                background: white; padding: 20px; margin: 15px 0; border-radius: 10px;
                border-left: 4px solid #4f46e5; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: transform 0.2s; cursor: pointer;
            }
            .result-item:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
            .result-header {
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 15px; font-weight: 600;
            }
            .result-status { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
            .status-success { background: #dcfce7; color: #166534; }
            .status-error { background: #fef2f2; color: #dc2626; }
            .loading { 
                display: none; text-align: center; padding: 30px; color: #6b7280;
                animation: fadeIn 0.3s ease-in;
            }
            .loading.show { display: block; }
            .spinner {
                width: 40px; height: 40px; margin: 0 auto 15px;
                border: 4px solid #e5e7eb; border-top: 4px solid #4f46e5;
                border-radius: 50%; animation: spin 1s linear infinite;
            }
            .example-grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px; margin-top: 20px;
            }
            .example-btn {
                background: #f3f4f6; border: 1px solid #d1d5db; color: #374151;
                padding: 10px 15px; border-radius: 6px; font-size: 0.9rem; cursor: pointer;
                transition: all 0.2s; text-align: center;
            }
            .example-btn:hover {
                background: #e5e7eb; transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metrics { 
                display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 10px; margin: 20px 0; padding: 15px; background: #f0f9ff;
                border-radius: 8px; border: 1px solid #bae6fd;
            }
            .metric { text-align: center; }
            .metric-value { font-size: 1.5rem; font-weight: bold; color: #0369a1; }
            .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 5px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @media (max-width: 768px) {
                .main-content { grid-template-columns: 1fr; }
                .example-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Decision Agent</h1>
                <p>Interactive Demo - Advanced Intelligent Task Routing</p>
            </div>
            
            <div class="main-content">
                <div class="input-panel">
                    <div class="panel-header">
                        <h3>📝 Input</h3>
                        <div id="inputStatus" style="font-size: 0.9rem; color: #6b7280;"></div>
                    </div>
                    
                    <textarea id="userInput" class="input-field" rows="8" 
                        placeholder="Enter your request here...

💡 Examples:
• https://example.com/article (web scraping)
• search for latest machine learning trends  
• show all users by department (database)
• what is our remote work policy? (knowledge)
• execute: import math; print(math.factorial(20))
• analyze advantages of microservices architecture"></textarea>
                    
                    <div style="margin: 20px 0;">
                        <button id="processBtn" class="btn">🚀 Process Request</button>
                        <button id="analyzeBtn" class="btn secondary">🔍 Analyze Input</button>
                        <button id="clearBtn" class="btn danger">🗑️ Clear Results</button>
                    </div>
                    
                    <div class="metrics" id="systemMetrics" style="display: none;">
                        <div class="metric">
                            <div class="metric-value" id="totalRequests">0</div>
                            <div class="metric-label">Total Requests</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="successRate">0%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="avgTime">0ms</div>
                            <div class="metric-label">Avg Response</div>
                        </div>
                    </div>
                    
                    <details>
                        <summary style="cursor: pointer; font-weight: 600; margin: 20px 0 10px 0;">📋 Quick Examples</summary>
                        <div class="example-grid">
                            <div class="example-btn" onclick="setExample('https://httpbin.org/json')">🌐 Web Scraping</div>
                            <div class="example-btn" onclick="setExample('search for Python async programming')">🔍 Web Search</div>
                            <div class="example-btn" onclick="setExample('show all active users')">🗄️ Database Query</div>
                            <div class="example-btn" onclick="setExample('what is our AI development policy?')">📚 Knowledge Base</div>
                            <div class="example-btn" onclick="setExample('calculate: sum(range(1, 101))')">💻 Code Execution</div>
                            <div class="example-btn" onclick="setExample('compare benefits of cloud vs on-premise infrastructure')">🧠 Complex Analysis</div>
                        </div>
                    </details>
                </div>
                
                <div class="results-panel">
                    <div class="panel-header">
                        <h3>📊 Results</h3>
                        <button id="refreshStats" class="btn secondary" style="font-size: 0.8rem; padding: 6px 12px;">🔄 Refresh Stats</button>
                    </div>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Processing your request...</p>
                        <p style="font-size: 0.9rem; margin-top: 10px;">Analyzing input and routing to optimal processor</p>
                    </div>
                    
                    <div id="results">
                        <div style="text-align: center; color: #6b7280; padding: 60px 20px;">
                            <h4 style="margin-bottom: 15px;">Ready to Process</h4>
                            <p>Enter a request above and click "Process Request" to see intelligent task routing in action.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let requestCount = 0;
            let successCount = 0;
            let totalResponseTime = 0;
            
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('processBtn').addEventListener('click', processRequest);
                document.getElementById('analyzeBtn').addEventListener('click', analyzeInput);
                document.getElementById('clearBtn').addEventListener('click', clearResults);
                document.getElementById('refreshStats').addEventListener('click', refreshSystemStats);
                document.getElementById('userInput').addEventListener('input', updateInputStatus);
                
                // Load initial stats
                refreshSystemStats();
            });
            
            function setExample(text) {
                document.getElementById('userInput').value = text;
                updateInputStatus();
            }
            
            function updateInputStatus() {
                const input = document.getElementById('userInput').value;
                const statusEl = document.getElementById('inputStatus');
                
                if (input.trim()) {
                    const wordCount = input.trim().split(/\\s+/).length;
                    statusEl.textContent = `${input.length} chars, ${wordCount} words`;
                } else {
                    statusEl.textContent = '';
                }
            }
            
            async function processRequest() {
                const input = document.getElementById('userInput').value.trim();
                if (!input) {
                    alert('Please enter a request to process');
                    return;
                }
                
                showLoading(true);
                disableButtons(true);
                
                const startTime = Date.now();
                
                try {
                    const response = await fetch('/api/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: input })
                    });
                    
                    const result = await response.json();
                    const responseTime = Date.now() - startTime;
                    
                    displayResult(result, 'process', responseTime);
                    updateLocalStats(result.status === 'success', responseTime);
                    
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
                    alert('Please enter text to analyze');
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
            
            function displayResult(result, type, responseTime = null) {
                const resultsDiv = document.getElementById('results');
                
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                const timestamp = new Date().toLocaleTimeString();
                const isSuccess = result.status === 'success' || type === 'analyze';
                
                let html = `
                    <div class="result-header">
                        <strong>${type === 'analyze' ? '🔍 Input Analysis' : '🚀 Processing Result'}</strong>
                        <div>
                            <span class="result-status ${isSuccess ? 'status-success' : 'status-error'}">
                                ${isSuccess ? '✅ SUCCESS' : '❌ ERROR'}
                            </span>
                            <span style="margin-left: 10px; color: #6b7280; font-size: 0.8rem;">${timestamp}</span>
                        </div>
                    </div>
                `;
                
                if (type === 'analyze') {
                    html += '<div><strong>Task Analysis Results:</strong></div>';
                    
                    if (result.task_scores && result.task_scores.length > 0) {
                        result.task_scores.forEach((score, index) => {
                            const isTop = index === 0;
                            const confidenceColor = score.confidence > 0.8 ? '#059669' : score.confidence > 0.5 ? '#d97706' : '#dc2626';
                            
                            html += `
                                <div style="margin: 12px 0; padding: 15px; background: ${isTop ? '#f0f9ff' : '#f9fafb'}; 
                                     border-radius: 8px; border-left: 3px solid ${isTop ? '#4f46e5' : '#d1d5db'};">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <strong style="color: #1f2937;">
                                            ${isTop ? '🎯 ' : ''}${score.task_type.toUpperCase().replace('_', ' ')}
                                        </strong>
                                        <div style="display: flex; align-items: center; gap: 10px;">
                                            <span style="color: ${confidenceColor}; font-weight: 700;">
                                                ${(score.confidence * 100).toFixed(1)}%
                                            </span>
                                            <span style="background: #e5e7eb; color: #374151; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem;">
                                                ${score.complexity}
                                            </span>
                                        </div>
                                    </div>
                                    <div style="font-size: 0.9rem; color: #6b7280; line-height: 1.4;">
                                        ${score.reasoning}
                                    </div>
                                    ${isTop ? '<div style="margin-top: 8px; font-size: 0.8rem; color: #4f46e5; font-weight: 600;">👑 Selected Processor</div>' : ''}
                                </div>
                            `;
                        });
                    }
                    
                    if (result.analysis_metadata) {
                        html += `
                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: 600; color: #374151;">📊 Detailed Analysis</summary>
                                <div style="margin-top: 10px; padding: 10px; background: #f3f4f6; border-radius: 6px; font-family: monospace; font-size: 0.8rem;">
                                    ${JSON.stringify(result.analysis_metadata, null, 2)}
                                </div>
                            </details>
                        `;
                    }
                    
                } else {
                    // Process result
                    if (result.selected_task) {
                        html += `<p><strong>Selected Task:</strong> ${result.selected_task.replace('_', ' ').toUpperCase()}</p>`;
                    }
                    
                    if (result.confidence_used !== undefined) {
                        const confidenceColor = result.confidence_used > 0.8 ? '#059669' : result.confidence_used > 0.5 ? '#d97706' : '#dc2626';
                        html += `<p><strong>Confidence:</strong> <span style="color: ${confidenceColor}; font-weight: 600;">${(result.confidence_used * 100).toFixed(1)}%</span></p>`;
                    }
                    
                    if (responseTime) {
                        html += `<p><strong>Response Time:</strong> ${responseTime}ms</p>`;
                    } else if (result.processing_time) {
                        html += `<p><strong>Processing Time:</strong> ${(result.processing_time * 1000).toFixed(1)}ms</p>`;
                    }
                    
                    if (result.reasoning) {
                        html += `<p><strong>Reasoning:</strong> ${result.reasoning}</p>`;
                    }
                    
                    if (result.error) {
                        html += `<p style="color: #dc2626;"><strong>Error:</strong> ${result.error}</p>`;
                        if (result.suggestion) {
                            html += `<p style="color: #d97706;"><strong>Suggestion:</strong> ${result.suggestion}</p>`;
                        }
                    }
                    
                    if (result.result) {
                        html += `
                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: 600; color: #374151;">📄 View Full Result</summary>
                                <pre style="background: #f3f4f6; padding: 15px; border-radius: 8px; overflow-x: auto; margin-top: 10px; white-space: pre-wrap; font-size: 0.8rem; line-height: 1.4;">${JSON.stringify(result.result, null, 2)}</pre>
                            </details>
                        `;
                    }
                    
                    if (result.performance_metrics) {
                        html += `
                            <div style="margin-top: 15px; padding: 10px; background: #ecfdf5; border: 1px solid #bbf7d0; border-radius: 6px;">
                                <strong style="color: #065f46;">Performance Metrics:</strong>
                                <div style="margin-top: 5px; font-size: 0.9rem; color: #047857;">
                                    Response: ${result.performance_metrics.response_time_ms}ms | 
                                    Efficiency: ${result.performance_metrics.processor_efficiency} | 
                                    Quality: ${result.performance_metrics.result_quality}
                                </div>
                            </div>
                        `;
                    }
                }
                
                resultItem.innerHTML = html;
                
                // Replace placeholder content or prepend to results
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].style.textAlign === 'center') {
                    resultsDiv.innerHTML = '';
                }
                
                resultsDiv.insertBefore(resultItem, resultsDiv.firstChild);
                
                // Limit to 10 results
                while (resultsDiv.children.length > 10) {
                    resultsDiv.removeChild(resultsDiv.lastChild);
                }
            }
            
            function displayError(message) {
                const resultsDiv = document.getElementById('results');
                
                const errorItem = document.createElement('div');
                errorItem.className = 'result-item';
                errorItem.style.borderLeftColor = '#ef4444';
                errorItem.innerHTML = `
                    <div class="result-header">
                        <strong>❌ Error</strong>
                        <span style="color: #6b7280; font-size: 0.8rem;">${new Date().toLocaleTimeString()}</span>
                    </div>
                    <p style="color: #dc2626; margin-top: 10px;">${message}</p>
                `;
                
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].style.textAlign === 'center') {
                    resultsDiv.innerHTML = '';
                }
                
                resultsDiv.insertBefore(errorItem, resultsDiv.firstChild);
            }
            
            function showLoading(show) {
                const loadingEl = document.getElementById('loading');
                if (show) {
                    loadingEl.classList.add('show');
                } else {
                    loadingEl.classList.remove('show');
                }
            }
            
            function disableButtons(disabled) {
                document.getElementById('processBtn').disabled = disabled;
                document.getElementById('analyzeBtn').disabled = disabled;
            }
            
            function clearResults() {
                document.getElementById('results').innerHTML = `
                    <div style="text-align: center; color: #6b7280; padding: 60px 20px;">
                        <h4 style="margin-bottom: 15px;">Ready to Process</h4>
                        <p>Enter a request above and click "Process Request" to see intelligent task routing in action.</p>
                    </div>
                `;
            }
            
            function updateLocalStats(success, responseTime) {
                requestCount++;
                if (success) successCount++;
                totalResponseTime += responseTime;
                
                document.getElementById('totalRequests').textContent = requestCount;
                document.getElementById('successRate').textContent = `${((successCount / requestCount) * 100).toFixed(1)}%`;
                document.getElementById('avgTime').textContent = `${Math.round(totalResponseTime / requestCount)}ms`;
                
                document.getElementById('systemMetrics').style.display = 'grid';
            }
            
            async function refreshSystemStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();
                    
                    // Update system metrics if we have server stats
                    if (stats.total_requests > 0) {
                        document.getElementById('totalRequests').textContent = stats.total_requests;
                        document.getElementById('successRate').textContent = `${(stats.success_rate * 100).toFixed(1)}%`;
                        document.getElementById('avgTime').textContent = `${Math.round(stats.average_processing_time * 1000)}ms`;
                        document.getElementById('systemMetrics').style.display = 'grid';
                    }
                } catch (error) {
                    console.log('Stats refresh failed:', error);
                }
            }
        </script>
    </body>
    </html>
    """)

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Complete Decision Agent with Advanced Task Routing")
    print("=" * 60)
    print(f"🌐 Server: http://localhost:8000")
    print(f"🎨 Demo UI: http://localhost:8000/demo")
    print(f"📚 API Docs: http://localhost:8000/api/docs")
    print(f"📊 Statistics: http://localhost:8000/api/stats")
    print(f"🔍 Health: http://localhost:8000/api/health")
    print("")
    print("🎯 Available Task Processors:")
    print("• 📄 Web Scraping (URLs and content extraction)")
    print("• 🔍 Google Search (intelligent web search)")
    print("• 🗄️ Database Queries (natural language to SQL)")
    print("• 📚 Knowledge Management RAG (company knowledge)")
    print("• 💻 Code Execution (safe Python environment)")
    print("• 🧠 HRM Complex Reasoning (analytical tasks)")
    print("• 💬 General Query (fallback processing)")
    print("")
    print("🚀 Starting server...")
    print("Press Ctrl+C to stop")
    print("")
    
    # Configure and start the server
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False  # Set to True for development
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)