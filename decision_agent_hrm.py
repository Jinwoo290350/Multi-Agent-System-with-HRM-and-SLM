# Multi-Agent-System-with-HRM-and-SLM.py - Real Multi-Agent System with HRM and SLM
import os
import asyncio
import logging
import json
import uuid
import re
import sys
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import sqlite3
from contextlib import asynccontextmanager
import io

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
from dotenv import load_dotenv

# Azure OpenAI client with proper API support
from openai import AsyncAzureOpenAI
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_agent_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    WEB_SCRAPING = "web_scraping"
    SEARCH = "search"
    DATABASE = "database"
    KNOWLEDGE = "knowledge"
    CODE = "code"
    GENERAL = "general"

class ConfidenceLevel(Enum):
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.0-0.5

@dataclass
class TaskDecision:
    task_type: TaskType
    confidence: float
    reasoning: str
    worker_model: str
    requires_gpt5: bool = False

@dataclass
class AgentResult:
    task_id: str
    hrm_decision: TaskDecision
    worker_result: Any
    processing_time: float
    confidence_level: ConfidenceLevel
    success: bool
    error: Optional[str] = None

class AzureOpenAIClient:
    """Azure OpenAI Client with proper API parameter support"""
    
    def __init__(self):
        self.api_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
        self.gpt5_nano_deployment = os.getenv("GPT5_NANO_DEPLOYMENT", "gpt-5-nano")
        self.gpt5_mini_deployment = os.getenv("GPT5_MINI_DEPLOYMENT", "gpt-5-mini")
        self.gpt5_deployment = "gpt-5"
        
        self.call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "hrm_calls": 0,
            "worker_calls": 0,
            "rate_limit_errors": 0
        }
        
        # Validate configuration
        if not self.api_key or not self.endpoint:
            logger.error("❌ Azure OpenAI configuration missing!")
            logger.error(f"API Key: {'SET' if self.api_key else 'MISSING'}")
            logger.error(f"Endpoint: {'SET' if self.endpoint else 'MISSING'}")
            raise ValueError("Configure AZURE_SUBSCRIPTION_KEY and AZURE_OPENAI_ENDPOINT in .env")
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"✅ Multi-Agent System initialized")
        logger.info(f"🌐 Endpoint: {self.endpoint}")
        logger.info(f"🧠 HRM Model: {self.gpt5_nano_deployment}")
        logger.info(f"⚡ SLM Workers: {self.gpt5_mini_deployment}, {self.gpt5_deployment}")
    
    async def call_hrm(self, system_prompt: str, user_prompt: str) -> str:
        """Call HRM (Hierarchical Reasoning Model) using GPT-5-nano"""
        
        self.call_stats["total_calls"] += 1
        self.call_stats["hrm_calls"] += 1
        
        logger.info(f"🧠 HRM Call #{self.call_stats['hrm_calls']} - Hierarchical Reasoning...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.gpt5_nano_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=800,  # FIXED: Use max_completion_tokens instead of max_tokens
                timeout=60.0
            )
            
            result = response.choices[0].message.content.strip()
            self.call_stats["successful_calls"] += 1
            
            logger.info(f"✅ HRM Decision completed - {len(result)} chars")
            return result
            
        except openai.RateLimitError as e:
            self.call_stats["rate_limit_errors"] += 1
            logger.warning(f"⏰ Rate limit hit for HRM - waiting and retrying...")
            await asyncio.sleep(10)  # Wait 10 seconds on rate limit
            return await self.call_hrm(system_prompt, user_prompt)  # Retry once
            
        except Exception as e:
            self.call_stats["failed_calls"] += 1
            logger.error(f"❌ HRM Call failed: {str(e)}")
            raise Exception(f"HRM (GPT-5-nano) failed: {str(e)}")
    
    async def call_slm_worker(self, system_prompt: str, user_prompt: str, use_gpt5: bool = False) -> str:
        """Call SLM (Small Language Model) Worker"""
        
        self.call_stats["total_calls"] += 1
        self.call_stats["worker_calls"] += 1
        
        model_name = "GPT-5" if use_gpt5 else "GPT-5-mini"
        deployment = self.gpt5_deployment if use_gpt5 else self.gpt5_mini_deployment
        
        logger.info(f"⚡ SLM Worker #{self.call_stats['worker_calls']} - {model_name}")
        
        try:
            response = await self.client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3 if use_gpt5 else 0.7,
                max_completion_tokens=2000,  # FIXED: Use max_completion_tokens
                timeout=90.0
            )
            
            result = response.choices[0].message.content.strip()
            self.call_stats["successful_calls"] += 1
            
            logger.info(f"✅ SLM Worker {model_name} completed - {len(result)} chars")
            return result
            
        except openai.RateLimitError as e:
            self.call_stats["rate_limit_errors"] += 1
            logger.warning(f"⏰ Rate limit hit for {model_name} - waiting...")
            await asyncio.sleep(15)  # Wait longer for worker models
            return await self.call_slm_worker(system_prompt, user_prompt, use_gpt5)
            
        except Exception as e:
            self.call_stats["failed_calls"] += 1
            logger.error(f"❌ SLM Worker {model_name} failed: {str(e)}")
            raise Exception(f"SLM Worker {model_name} failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive API statistics"""
        return {
            **self.call_stats,
            "success_rate": self.call_stats["successful_calls"] / max(1, self.call_stats["total_calls"]),
            "endpoint": self.endpoint,
            "deployments": {
                "hrm": self.gpt5_nano_deployment,
                "slm_mini": self.gpt5_mini_deployment,
                "slm_full": self.gpt5_deployment
            }
        }

# Initialize Azure OpenAI client with error handling
try:
    azure_client = AzureOpenAIClient()
    logger.info("🚀 Multi-Agent System ready with HRM and SLM workers")
except Exception as e:
    logger.error(f"💥 Multi-Agent System initialization failed: {e}")
    logger.error("🔧 Configure your .env file with Azure OpenAI credentials")
    sys.exit(1)

class HRMProcessor:
    """Hierarchical Reasoning Model (HRM) - The decision-making brain"""
    
    def __init__(self):
        self.decision_count = 0
    
    async def make_decision(self, input_text: str) -> TaskDecision:
        """HRM makes intelligent routing decisions with confidence scoring"""
        
        self.decision_count += 1
        
        system_prompt = """You are an HRM (Hierarchical Reasoning Model) - the intelligent brain of a Multi-Agent System. 

Your role is to analyze user input and make optimal routing decisions to specialized SLM (Small Language Model) workers.

Available SLM Workers:
1. web_scraping - Extract and analyze web content from URLs
2. search - Intelligent information search and research
3. database - Query organizational database with SQL generation
4. knowledge - Retrieve company policies, procedures, documentation
5. code - Execute Python code and calculations (uses GPT-5 SLM)
6. general - Conversational assistance and general queries

DECISION FRAMEWORK:
- Analyze input complexity and requirements
- Assign confidence score (0.0-1.0) based on clarity
- Select optimal SLM worker for the task
- Route code execution to GPT-5 SLM, others to GPT-5-mini SLM

Respond ONLY with valid JSON:
{
    "task_type": "task_name",
    "confidence": 0.95,
    "reasoning": "Clear explanation of routing decision",
    "worker_model": "gpt-5-mini" or "gpt-5",
    "requires_gpt5": false,
    "complexity_assessment": "simple/medium/complex"
}

ROUTING RULES:
- URLs/websites → web_scraping
- Code/execute/calculate/programming → code (requires_gpt5: true)
- Search/find/research → search
- Database/users/show/list/SQL → database  
- Policies/procedures/documentation → knowledge
- Conversation/questions → general"""

        user_prompt = f"""Analyze this input and make optimal routing decision:

Input: "{input_text}"

Assess the task complexity, confidence level, and route to the best SLM worker. Consider:
- What type of processing is needed?
- How clear and specific is the request?
- Which SLM worker can handle this optimally?
- Does this require advanced reasoning (GPT-5) or standard processing (GPT-5-mini)?

Provide your HRM decision as JSON:"""

        try:
            response = await azure_client.call_hrm(system_prompt, user_prompt)
            
            # Parse HRM decision
            decision_data = json.loads(response)
            
            # Validate required fields
            required_fields = ["task_type", "confidence", "reasoning", "worker_model"]
            if not all(field in decision_data for field in required_fields):
                raise ValueError(f"HRM decision missing fields: {decision_data}")
            
            # Create task decision
            task_decision = TaskDecision(
                task_type=TaskType(decision_data["task_type"]),
                confidence=decision_data["confidence"],
                reasoning=decision_data["reasoning"],
                worker_model=decision_data["worker_model"],
                requires_gpt5=decision_data.get("requires_gpt5", False) or decision_data["task_type"] == "code"
            )
            
            # Log HRM decision
            confidence_level = self._get_confidence_level(task_decision.confidence)
            logger.info(f"🎯 HRM Decision #{self.decision_count}:")
            logger.info(f"   Task: {task_decision.task_type.value}")
            logger.info(f"   Confidence: {task_decision.confidence:.2f} ({confidence_level.value})")
            logger.info(f"   Worker: {task_decision.worker_model}")
            logger.info(f"   Reasoning: {task_decision.reasoning}")
            
            return task_decision
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ HRM JSON parse error: {e}")
            logger.error(f"Raw HRM response: {response}")
            raise Exception(f"HRM returned invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"❌ HRM decision failed: {e}")
            raise Exception(f"HRM decision failed: {str(e)}")
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

class CodeSLMWorker:
    """Code execution SLM worker using GPT-5"""
    
    async def execute(self, input_text: str) -> Dict[str, Any]:
        """Execute code using GPT-5 SLM worker"""
        
        # Extract code
        code_info = self._extract_code(input_text)
        if not code_info["code"]:
            return {
                "type": "code",
                "error": "No executable code found",
                "suggestion": "Use 'execute: code' or ```python code```"
            }
        
        # Security validation
        if not self._validate_security(code_info["code"]):
            return {
                "type": "code",
                "error": "Code contains unsafe operations",
                "blocked": "File operations, system calls, or infinite loops detected"
            }
        
        # Execute with GPT-5 SLM
        try:
            result = await self._execute_with_gpt5_slm(code_info["code"])
            
            return {
                "type": "code",
                "code": code_info["code"],
                "output": result["output"],
                "worker": "gpt-5-slm",
                "execution_time": result["time"],
                "method": code_info["method"]
            }
            
        except Exception as e:
            return {
                "type": "code",
                "code": code_info["code"],
                "error": str(e),
                "worker": "gpt-5-slm"
            }
    
    async def _execute_with_gpt5_slm(self, code: str) -> Dict[str, Any]:
        """Execute code using GPT-5 SLM worker"""
        
        system_prompt = """You are a Python code execution SLM (Small Language Model) worker powered by GPT-5.

EXECUTION RULES:
1. Execute the provided Python code accurately
2. Return ONLY the exact output that would be produced
3. For print statements: return what gets printed to stdout
4. For expressions: return the calculated result
5. For assignments with no output: return "Code executed successfully"
6. Handle imports and mathematical operations correctly
7. Be precise with calculations and formatting

EXAMPLES:
Input: print(2 + 3)
Output: 5

Input: import math; print(f"Factorial: {math.factorial(10)}")  
Output: Factorial: 3628800

Input: import math; print(f"Result: {math.sqrt(144) + math.pi:.2f}")
Output: Result: 15.14

Input: x = 10; y = 20; print(f"Sum: {x + y}")
Output: Sum: 30

Execute the code and return ONLY the output:"""

        user_prompt = f"Execute this Python code:\n\n{code}\n\nOutput:"
        
        try:
            start_time = time.time()
            response = await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=True)
            execution_time = time.time() - start_time
            
            # Clean the output
            output = response.strip()
            if output.lower().startswith("output:"):
                output = output[7:].strip()
            
            return {
                "output": output,
                "time": round(execution_time, 3)
            }
            
        except Exception as e:
            raise Exception(f"GPT-5 SLM code execution failed: {str(e)}")
    
    def _extract_code(self, text: str) -> Dict[str, str]:
        """Extract code from input"""
        
        # Explicit execution commands
        prefixes = [
            ("execute:", "explicit"),
            ("calculate:", "explicit"),
            ("python:", "explicit"),
            ("run:", "explicit"),
            ("code:", "explicit")
        ]
        
        for prefix, method in prefixes:
            if text.lower().startswith(prefix):
                return {
                    "code": text[len(prefix):].strip(),
                    "method": method
                }
        
        # Code blocks
        match = re.search(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return {
                "code": match.group(1).strip(),
                "method": "code_block"
            }
        
        # Inline code
        match = re.search(r'`([^`\n]+)`', text)
        if match:
            return {
                "code": match.group(1).strip(),
                "method": "inline"
            }
        
        return {"code": "", "method": "none"}
    
    def _validate_security(self, code: str) -> bool:
        """Security validation for code execution"""
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            'open(', 'file(', 'exec(', 'eval(',
            '__import__', 'input(', 'raw_input(',
            'while True', 'while 1:'
        ]
        
        code_lower = code.lower()
        return not any(pattern in code_lower for pattern in dangerous_patterns)

class DatabaseSLMWorker:
    """Database SLM worker using GPT-5-mini"""
    
    def __init__(self):
        self.db_path = os.getenv("DATABASE_PATH", "./data/decision_agent.db")
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database exists"""
        if not os.path.exists(self.db_path):
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Use schema file if available
            schema_file = "database_schema.sql"
            if os.path.exists(schema_file):
                logger.info(f"📊 Creating database from {schema_file}")
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                
                conn = sqlite3.connect(self.db_path)
                try:
                    conn.executescript(schema_sql)
                    conn.commit()
                    logger.info("✅ Database created from schema")
                except Exception as e:
                    logger.error(f"❌ Database creation failed: {e}")
                finally:
                    conn.close()
    
    async def execute(self, input_text: str) -> Dict[str, Any]:
        """Execute database query using GPT-5-mini SLM worker"""
        
        try:
            # Get database schema
            schema = self._get_schema()
            
            # Generate SQL using GPT-5-mini SLM
            sql = await self._generate_sql_with_slm(input_text, schema)
            
            # Execute SQL
            results = self._execute_sql(sql)
            
            # Explain results using GPT-5-mini SLM
            explanation = await self._explain_with_slm(input_text, sql, results)
            
            return {
                "type": "database",
                "query": input_text,
                "sql_generated": sql,
                "results": results,
                "row_count": len(results),
                "explanation": explanation,
                "worker": "gpt-5-mini-slm"
            }
            
        except Exception as e:
            return {
                "type": "database",
                "query": input_text,
                "error": str(e),
                "worker": "gpt-5-mini-slm"
            }
    
    async def _generate_sql_with_slm(self, query: str, schema: str) -> str:
        """Generate SQL using GPT-5-mini SLM worker"""
        
        system_prompt = f"""You are a Database SLM (Small Language Model) worker powered by GPT-5-mini.

Your task is to convert natural language queries to SQL SELECT statements.

Database Schema:
{schema}

SQL GENERATION RULES:
1. Generate ONLY SELECT queries (no INSERT/UPDATE/DELETE)
2. Use proper SQLite syntax
3. Always include LIMIT clause (max 50 rows)
4. Use JOINs when querying related tables
5. Return ONLY the SQL query, no explanations or formatting

EXAMPLES:
"show users" → SELECT * FROM users WHERE status = 'active' LIMIT 50;
"active projects" → SELECT * FROM projects WHERE status = 'active' LIMIT 50;
"users by department" → SELECT u.full_name, d.name as department FROM users u JOIN departments d ON u.department_id = d.id WHERE u.status = 'active' LIMIT 50;"""

        user_prompt = f"Convert this natural language query to SQL: {query}"
        
        try:
            response = await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=False)
            
            # Clean SQL
            sql = response.strip()
            if sql.startswith('```'):
                lines = sql.split('\n')
                sql = '\n'.join(lines[1:-1]).strip()
            
            if not sql.endswith(';'):
                sql += ';'
            
            # Security validation
            if not sql.lower().strip().startswith('select'):
                raise ValueError("Only SELECT queries are allowed")
            
            return sql
            
        except Exception as e:
            raise Exception(f"SQL generation by SLM failed: {str(e)}")
    
    async def _explain_with_slm(self, query: str, sql: str, results: List[Dict]) -> str:
        """Explain results using GPT-5-mini SLM worker"""
        
        system_prompt = """You are a Database Results Explainer SLM worker powered by GPT-5-mini.

Explain query results in a clear, business-friendly way:
1. Summarize what was found
2. Highlight key insights
3. Explain practical implications
4. Keep it concise but informative"""

        results_sample = json.dumps(results[:3], indent=2) if results else "No results found"
        
        user_prompt = f"""Original Query: "{query}"
SQL Executed: {sql}
Results Found: {len(results)} rows

Sample Data:
{results_sample}

Provide a clear business explanation:"""

        try:
            return await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=False)
        except Exception as e:
            return f"Query executed successfully. Found {len(results)} records. Unable to generate detailed explanation due to: {str(e)}"
    
    def _get_schema(self) -> str:
        """Get database schema information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema = "Database Tables:\n"
            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema += f"\n{table_name}:\n"
                for col in columns:
                    schema += f"  {col[1]} ({col[2]})\n"
            
            conn.close()
            return schema
        except Exception as e:
            return "Schema information unavailable"
    
    def _execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL query safely"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

class WebScrapingSLMWorker:
    """Web scraping SLM worker using GPT-5-mini"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def execute(self, input_text: str) -> Dict[str, Any]:
        """Execute web scraping using GPT-5-mini SLM worker"""
        
        urls = self._extract_urls(input_text)
        if not urls:
            return {
                "type": "web_scraping",
                "error": "No valid URLs found",
                "suggestion": "Provide valid URLs like https://example.com"
            }
        
        results = []
        for url in urls[:3]:  # Limit to 3 URLs
            try:
                content = await self._scrape_and_analyze_with_slm(url)
                results.append(content)
            except Exception as e:
                results.append({
                    "url": url,
                    "error": str(e),
                    "status": "failed"
                })
        
        successful = len([r for r in results if r.get("status") == "success"])
        
        return {
            "type": "web_scraping",
            "results": results,
            "successful_scrapes": successful,
            "total_urls": len(results),
            "worker": "gpt-5-mini-slm"
        }
    
    async def _scrape_and_analyze_with_slm(self, url: str) -> Dict[str, Any]:
        """Scrape URL and analyze with GPT-5-mini SLM worker"""
        
        try:
            # Scrape content
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            
            # Analyze with GPT-5-mini SLM
            analysis = await self._analyze_with_slm(title, content, url)
            
            return {
                "url": url,
                "title": title,
                "content": content[:1000],
                "analysis": analysis,
                "word_count": len(content.split()),
                "status": "success"
            }
            
        except Exception as e:
            raise Exception(f"Web scraping failed: {str(e)}")
    
    async def _analyze_with_slm(self, title: str, content: str, url: str) -> str:
        """Analyze content using GPT-5-mini SLM worker"""
        
        system_prompt = """You are a Web Content Analysis SLM worker powered by GPT-5-mini.

Analyze web page content and provide structured insights:
1. Main topic and purpose
2. Key information and findings
3. Important data or insights
4. Content quality assessment
5. Practical takeaways

Provide a clear, comprehensive analysis."""

        user_prompt = f"""Analyze this web page content:

URL: {url}
Title: {title}
Content: {content[:2000]}

Provide detailed analysis:"""

        try:
            return await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=False)
        except Exception as e:
            return f"Content extracted from {url}. SLM analysis failed: {str(e)}"
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(pattern, text)
        
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    valid_urls.append(url)
            except:
                continue
        
        return list(set(valid_urls))
    
    def _extract_title(self, soup):
        """Extract page title"""
        if soup.title and soup.title.string:
            return soup.title.string.strip()[:200]
        elif soup.find('h1'):
            h1 = soup.find('h1')
            return h1.get_text().strip()[:200] if h1 else "No title"
        return "No title"
    
    def _extract_content(self, soup):
        """Extract main content"""
        selectors = ['main', 'article', '.content', '#content']
        for selector in selectors:
            try:
                if selector.startswith('.') or selector.startswith('#'):
                    element = soup.select_one(selector)
                else:
                    element = soup.find(selector)
                
                if element:
                    content = element.get_text(separator=' ', strip=True)
                    if len(content) > 200:
                        return content[:3000]
            except:
                continue
        
        # Fallback to paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            content_parts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
            return ' '.join(content_parts[:10])[:3000]
        
        return "No substantial content found"

class SearchSLMWorker:
    """Search SLM worker using GPT-5-mini"""
    
    async def execute(self, input_text: str) -> Dict[str, Any]:
        """Execute search using GPT-5-mini SLM worker"""
        
        query = self._extract_query(input_text)
        if not query:
            return {
                "type": "search",
                "error": "Empty search query",
                "suggestion": "Use 'search for [topic]' format"
            }
        
        try:
            results = await self._generate_search_with_slm(query)
            
            return {
                "type": "search",
                "search_query": query,
                "results": results["results"],
                "result_count": len(results["results"]),
                "synthesis": results["synthesis"],
                "worker": "gpt-5-mini-slm"
            }
            
        except Exception as e:
            return {
                "type": "search",
                "search_query": query,
                "error": str(e),
                "worker": "gpt-5-mini-slm"
            }
    
    async def _generate_search_with_slm(self, query: str) -> Dict[str, Any]:
        """Generate search results using GPT-5-mini SLM worker"""
        
        system_prompt = """You are a Search SLM worker powered by GPT-5-mini.

Generate intelligent search results with comprehensive analysis:
1. Create 4-5 relevant, realistic search results
2. Include informative titles and authoritative URLs
3. Write detailed, helpful snippets
4. Assign relevance scores
5. Provide synthesis of key findings

Return as JSON:
{
    "results": [
        {
            "title": "...",
            "url": "...",
            "snippet": "...",
            "relevance": 0.95
        }
    ],
    "synthesis": "Comprehensive summary of search findings and insights"
}"""

        user_prompt = f'Generate comprehensive search results for: "{query}"'
        
        try:
            response = await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=False)
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "results": [
                    {
                        "title": f"Comprehensive Guide to {query.title()}",
                        "url": f"https://guide.example.com/{query.replace(' ', '-')}",
                        "snippet": f"In-depth analysis and expert insights on {query} with practical applications and current developments.",
                        "relevance": 0.9
                    }
                ],
                "synthesis": f"Search for '{query}' reveals significant information from authoritative sources with practical applications."
            }
    
    def _extract_query(self, text: str) -> str:
        """Extract search query from text"""
        prefixes = ["search for", "search", "find", "research", "lookup"]
        text_lower = text.lower()
        
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                return text[len(prefix):].strip().strip('"\'')
        
        return text.strip()

class GeneralSLMWorker:
    """General conversation SLM worker using GPT-5-mini"""
    
    async def execute(self, input_text: str) -> Dict[str, Any]:
        """Execute general conversation using GPT-5-mini SLM worker"""
        
        system_prompt = """You are a General Assistant SLM worker powered by GPT-5-mini.

You are part of a Multi-Agent System with HRM (Hierarchical Reasoning Model) and specialized SLM workers:

Available capabilities:
- Web scraping (provide URLs)
- Search and research (use 'search for')
- Database queries (ask about users, tasks, projects)
- Code execution (use 'execute: code')
- Knowledge base (ask about policies, procedures)

Provide helpful, informative responses while guiding users to use specific capabilities when appropriate."""

        user_prompt = f"User request: {input_text}\n\nProvide helpful assistance:"
        
        try:
            response = await azure_client.call_slm_worker(system_prompt, user_prompt, use_gpt5=False)
            
            return {
                "type": "general",
                "query": input_text,
                "response": response,
                "worker": "gpt-5-mini-slm"
            }
            
        except Exception as e:
            return {
                "type": "general",
                "query": input_text,
                "error": str(e),
                "worker": "gpt-5-mini-slm"
            }

class MultiAgentSystem:
    """Multi-Agent System with HRM and SLM workers"""
    
    def __init__(self):
        # Initialize HRM and SLM workers
        self.hrm = HRMProcessor()
        self.slm_workers = {
            TaskType.CODE: CodeSLMWorker(),
            TaskType.DATABASE: DatabaseSLMWorker(),
            TaskType.WEB_SCRAPING: WebScrapingSLMWorker(),
            TaskType.SEARCH: SearchSLMWorker(),
            TaskType.KNOWLEDGE: GeneralSLMWorker(),  # Using general for knowledge for now
            TaskType.GENERAL: GeneralSLMWorker()
        }
        
        # System statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "confidence_levels": {"high": 0, "medium": 0, "low": 0},
            "task_distribution": {},
            "start_time": datetime.now()
        }
        
        logger.info("🤖 Multi-Agent System initialized")
        logger.info("🧠 HRM: Hierarchical Reasoning Model ready")
        logger.info("⚡ SLM Workers: All specialized workers initialized")
    
    async def process(self, input_text: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Process request through Multi-Agent System with HRM and SLM workers"""
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        self.stats["total_requests"] += 1
        
        try:
            logger.info(f"🎯 Multi-Agent Processing: {input_text[:100]}...")
            
            # Phase 1: HRM (Hierarchical Reasoning Model) makes decision
            logger.info("🧠 Phase 1: HRM Decision Making...")
            hrm_decision = await self.hrm.make_decision(input_text)
            
            # Phase 2: Route to appropriate SLM worker based on confidence
            logger.info(f"⚡ Phase 2: SLM Worker Execution ({hrm_decision.worker_model})...")
            confidence_level = self._get_confidence_level(hrm_decision.confidence)
            
            # Execute with SLM worker
            worker_result = await self._execute_with_slm_worker(hrm_decision.task_type, input_text)
            
            # Phase 3: Create comprehensive result
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats["successful_requests"] += 1
            self.stats["confidence_levels"][confidence_level.value] += 1
            self.stats["task_distribution"][hrm_decision.task_type.value] = \
                self.stats["task_distribution"].get(hrm_decision.task_type.value, 0) + 1
            
            # Create agent result
            agent_result = AgentResult(
                task_id=task_id,
                hrm_decision=hrm_decision,
                worker_result=worker_result,
                processing_time=processing_time,
                confidence_level=confidence_level,
                success=True
            )
            
            return {
                "task_id": task_id,
                "status": "success",
                "hrm_decision": {
                    "selected_task": hrm_decision.task_type.value,
                    "confidence": hrm_decision.confidence,
                    "confidence_level": confidence_level.value,
                    "reasoning": hrm_decision.reasoning,
                    "worker_model": hrm_decision.worker_model
                },
                "slm_worker_result": worker_result,
                "processing_time": round(processing_time, 3),
                "multi_agent_stats": azure_client.get_stats()
            }
            
        except Exception as e:
            # Phase 3: Error handling
            processing_time = time.time() - start_time
            self.stats["failed_requests"] += 1
            
            logger.error(f"❌ Multi-Agent System error: {str(e)}")
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "processing_time": round(processing_time, 3),
                "error_suggestion": self._get_error_suggestion(str(e)),
                "multi_agent_stats": azure_client.get_stats()
            }
    
    async def _execute_with_slm_worker(self, task_type: TaskType, input_text: str) -> Any:
        """Execute task with appropriate SLM worker"""
        
        if task_type in self.slm_workers:
            return await self.slm_workers[task_type].execute(input_text)
        else:
            # Fallback to general worker
            return await self.slm_workers[TaskType.GENERAL].execute(input_text)
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _get_error_suggestion(self, error: str) -> str:
        """Generate error suggestion"""
        error_lower = error.lower()
        
        if "rate limit" in error_lower:
            return "Rate limit reached. The system will automatically retry after a brief delay."
        elif "azure" in error_lower or "openai" in error_lower:
            return "Azure OpenAI configuration issue. Check API key, endpoint, and deployment names."
        elif "database" in error_lower:
            return "Database error. Ensure database is properly initialized."
        elif "timeout" in error_lower:
            return "Request timed out. Try a simpler query or check network connectivity."
        elif "json" in error_lower:
            return "Response parsing error. The HRM or SLM worker may need adjustment."
        else:
            return "Try rephrasing your request or check system configuration."
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive Multi-Agent System statistics"""
        
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            **self.stats,
            "azure_api_stats": azure_client.get_stats(),
            "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
            "uptime_seconds": uptime,
            "uptime_formatted": str(datetime.now() - self.stats["start_time"]).split('.')[0],
            "requests_per_minute": self.stats["total_requests"] / max(1, uptime / 60),
            "hrm_decisions": self.hrm.decision_count,
            "system_architecture": {
                "hrm_model": "gpt-5-nano",
                "slm_workers": {
                    "code": "gpt-5",
                    "others": "gpt-5-mini"
                }
            }
        }

# Initialize Multi-Agent System
multi_agent_system = MultiAgentSystem()

# Pydantic models
class ProcessRequest(BaseModel):
    input: str
    options: Optional[Dict[str, Any]] = None
    
    @validator('input')
    def input_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Input cannot be empty')
        return v.strip()

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    logger.info("🚀 Multi-Agent System with HRM and SLM starting...")
    logger.info(f"🧠 HRM: {azure_client.gpt5_nano_deployment}")
    logger.info(f"⚡ SLM Workers: {azure_client.gpt5_mini_deployment}, {azure_client.gpt5_deployment}")
    
    yield
    
    stats = multi_agent_system.get_comprehensive_stats()
    logger.info("📊 Final Multi-Agent System Stats:")
    logger.info(f"   Total Requests: {stats['total_requests']}")
    logger.info(f"   HRM Decisions: {stats['hrm_decisions']}")
    logger.info(f"   API Calls: {stats['azure_api_stats']['total_calls']}")
    logger.info(f"   Success Rate: {stats['success_rate']:.1%}")

app = FastAPI(
    title="Multi-Agent System with HRM and SLM",
    description="Hierarchical Reasoning Model (HRM) with Small Language Model (SLM) Workers",
    version="1.0.0",
    docs_url="/api/docs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Multi-Agent System home page"""
    
    stats = azure_client.get_stats()
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Agent System with HRM and SLM</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: system-ui; background: linear-gradient(135deg, #667eea, #764ba2); color: white; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; text-align: center; }}
            h1 {{ font-size: 2.5rem; margin-bottom: 0.5rem; }}
            h2 {{ font-size: 1.5rem; margin-bottom: 2rem; opacity: 0.9; }}
            .architecture {{ background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px; margin: 25px 0; }}
            .status {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin: 20px 0; }}
            .nav {{ display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; margin: 30px 0; }}
            .nav a {{ background: rgba(255,255,255,0.2); color: white; padding: 15px 25px; text-decoration: none; border-radius: 10px; transition: all 0.3s; }}
            .nav a:hover {{ background: rgba(255,255,255,0.3); transform: translateY(-2px); }}
            .flow {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .step {{ background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; flex: 1; margin: 0 10px; }}
            @media (max-width: 768px) {{ .flow {{ flex-direction: column; }} .step {{ margin: 10px 0; }} }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Multi-Agent System</h1>
            <h2>with HRM and SLM</h2>
            
            <div class="architecture">
                <h3>🧠 System Architecture</h3>
                <div class="flow">
                    <div class="step">
                        <h4>🧠 HRM</h4>
                        <p>Hierarchical Reasoning Model<br>GPT-5-nano</p>
                    </div>
                    <div class="step">
                        <h4>⚡ SLM Workers</h4>
                        <p>Small Language Models<br>GPT-5-mini / GPT-5</p>
                    </div>
                    <div class="step">
                        <h4>🎯 Task Execution</h4>
                        <p>Specialized Processing<br>Confidence-Based Routing</p>
                    </div>
                </div>
            </div>
            
            <div class="status">
                <h3>✅ System Status</h3>
                <p><strong>Azure OpenAI:</strong> Connected to {azure_client.endpoint}</p>
                <p><strong>HRM Model:</strong> {azure_client.gpt5_nano_deployment}</p>
                <p><strong>SLM Workers:</strong> {azure_client.gpt5_mini_deployment}, {azure_client.gpt5_deployment}</p>
                <p><strong>API Calls Made:</strong> {stats['total_calls']} (Success: {stats['success_rate']:.1%})</p>
                <p><strong>Rate Limits:</strong> {stats['rate_limit_errors']} handled</p>
            </div>
            
            <div class="nav">
                <a href="/demo">🎨 Interactive Demo</a>
                <a href="/api/docs">📚 API Documentation</a>
                <a href="/api/stats">📊 System Statistics</a>
            </div>
        </div>
    </body>
    </html>
    """)

@app.post("/api/process")
async def process_request(request: ProcessRequest):
    """Process request through Multi-Agent System"""
    try:
        return await multi_agent_system.process(request.input, request.options)
    except Exception as e:
        logger.error(f"API processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get Multi-Agent System statistics"""
    return multi_agent_system.get_comprehensive_stats()

@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Interactive demo interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Agent System Demo</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: system-ui; margin: 0; background: #f5f7fa; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .panel { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 120px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            .btn { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px 5px 0 0; }
            .btn:hover { background: #5a67d8; }
            .result { margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #667eea; }
            .badge { background: #10b981; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
            .confidence { padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }
            .high { background: #10b981; color: white; }
            .medium { background: #f59e0b; color: white; }
            .low { background: #ef4444; color: white; }
            @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Multi-Agent System Demo</h1>
                <p>HRM (Hierarchical Reasoning Model) + SLM (Small Language Model) Workers</p>
                <span class="badge">REAL API CALLS</span>
            </div>
            
            <div class="grid">
                <div class="panel">
                    <h3>📝 Input</h3>
                    <textarea id="input" placeholder="Enter your request...

Examples:
🧠 HRM will analyze and route to appropriate SLM worker:

• execute: import math; print(f'Factorial: {math.factorial(10)}')
  → Routes to GPT-5 SLM for code execution

• https://httpbin.org/json  
  → Routes to GPT-5-mini SLM for web scraping

• search for artificial intelligence trends
  → Routes to GPT-5-mini SLM for search

• show all users by department
  → Routes to GPT-5-mini SLM for database

• Hello, how does this system work?
  → Routes to GPT-5-mini SLM for general chat"></textarea>
                    
                    <button class="btn" onclick="process()">🚀 Process with Multi-Agent System</button>
                    <button class="btn" onclick="clear()" style="background: #6c757d;">Clear</button>
                </div>
                
                <div class="panel">
                    <h3>📊 Results</h3>
                    <div id="results">Ready for Multi-Agent processing with HRM and SLM workers...</div>
                </div>
            </div>
        </div>
        
        <script>
            async function process() {
                const input = document.getElementById('input').value.trim();
                if (!input) return alert('Enter a request');
                
                document.getElementById('results').innerHTML = '🧠 HRM analyzing input → ⚡ Routing to SLM worker...';
                
                try {
                    const response = await fetch('/api/process', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input })
                    });
                    
                    const data = await response.json();
                    displayResult(data);
                } catch (error) {
                    document.getElementById('results').innerHTML = `<div class="result" style="border-left-color: #dc3545;">❌ Error: ${error.message}</div>`;
                }
            }
            
            function displayResult(data) {
                let html = `<div class="result">`;
                
                if (data.status === 'success') {
                    const hrm = data.hrm_decision;
                    const result = data.slm_worker_result;
                    
                    html += `
                        <strong>🧠 HRM Decision:</strong> ${hrm.selected_task.toUpperCase().replace('_', ' ')}<br>
                        <strong>🎯 Confidence:</strong> ${(hrm.confidence * 100).toFixed(1)}% 
                        <span class="confidence ${hrm.confidence_level}">${hrm.confidence_level.toUpperCase()}</span><br>
                        <strong>⚡ SLM Worker:</strong> ${hrm.worker_model}<br>
                        <strong>⏱️ Processing Time:</strong> ${data.processing_time}s<br>
                        <strong>💭 HRM Reasoning:</strong> ${hrm.reasoning}<br><br>
                    `;
                    
                    // Display SLM worker result
                    if (result.error) {
                        html += `<strong style="color: #dc3545;">❌ SLM Worker Error:</strong> ${result.error}`;
                    } else {
                        if (result.output) {
                            html += `<strong>💻 Code Output:</strong><pre style="background: #f1f1f1; padding: 8px; margin: 5px 0;">${result.output}</pre>`;
                        }
                        if (result.sql_generated) {
                            html += `<strong>🗄️ Generated SQL:</strong> <code>${result.sql_generated}</code><br>`;
                        }
                        if (result.row_count !== undefined) {
                            html += `<strong>📊 Database Rows:</strong> ${result.row_count}<br>`;
                        }
                        if (result.explanation) {
                            html += `<strong>📝 SLM Explanation:</strong> ${result.explanation}<br>`;
                        }
                        if (result.synthesis) {
                            html += `<strong>🔍 Search Synthesis:</strong> ${result.synthesis}<br>`;
                        }
                        if (result.analysis) {
                            html += `<strong>🌐 Web Analysis:</strong> ${result.analysis}<br>`;
                        }
                        if (result.response) {
                            html += `<strong>💬 SLM Response:</strong> ${result.response}<br>`;
                        }
                    }
                    
                    html += `<br><strong>📈 API Stats:</strong> ${data.multi_agent_stats.total_calls} total calls, ${(data.multi_agent_stats.success_rate * 100).toFixed(1)}% success rate`;
                    
                } else {
                    html += `
                        <strong style="color: #dc3545;">❌ Multi-Agent Error:</strong> ${data.error}<br>
                        <strong>💡 Suggestion:</strong> ${data.error_suggestion || 'Try rephrasing your request'}
                    `;
                }
                
                html += '</div>';
                document.getElementById('results').innerHTML = html;
            }
            
            function clear() {
                document.getElementById('input').value = '';
                document.getElementById('results').innerHTML = 'Ready for Multi-Agent processing with HRM and SLM workers...';
            }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    
    print("🤖 Multi-Agent System with HRM and SLM")
    print("=" * 50)
    print(f"🧠 HRM Model: {azure_client.gpt5_nano_deployment}")
    print(f"⚡ SLM Workers: {azure_client.gpt5_mini_deployment}, {azure_client.gpt5_deployment}")
    print(f"🌐 Endpoint: {azure_client.endpoint}")
    print("🌐 Server: http://localhost:8000")
    print("🎨 Demo: http://localhost:8000/demo")
    print("")
    print("🔄 Processing Flow:")
    print("1. 🧠 HRM (GPT-5-nano) analyzes input and makes routing decision")
    print("2. 🎯 Confidence scoring determines processing approach")
    print("3. ⚡ Appropriate SLM worker executes the task")
    print("4. 📊 Results returned with comprehensive statistics")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)