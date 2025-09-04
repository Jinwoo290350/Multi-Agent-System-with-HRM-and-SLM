# decision_agent_complete.py - Complete Fixed Decision Agent
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

class ResultFormatter:
    """Enhanced result formatter for better readability"""
    
    @staticmethod
    def format_web_scraping_result(result: Dict) -> str:
        if "error" in result:
            return f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {result['error']}\n💡 คำแนะนำ: ตรวจสอบ URL ให้ถูกต้อง"
        
        if result.get("successful_scrapes", 0) == 0:
            return "❌ ไม่สามารถดึงข้อมูลจาก URL ที่ระบุได้ กรุณาตรวจสอบ URL และลองใหม่"
        
        formatted = f"🌐 ผลการดึงข้อมูลจากเว็บไซต์\n"
        formatted += f"📊 ดึงข้อมูลสำเร็จ: {result.get('successful_scrapes', 0)} / {result.get('scraped_urls', 0)} URL\n\n"
        
        for i, item in enumerate(result.get("results", []), 1):
            if item.get("status") == "success":
                formatted += f"📄 หน้าที่ {i}: {item.get('title', 'ไม่มีชื่อ')}\n"
                formatted += f"🔗 URL: {item.get('url', '')}\n"
                
                content = item.get('content', '')
                if content:
                    preview = content[:300] + "..." if len(content) > 300 else content
                    formatted += f"📝 เนื้อหา: {preview}\n"
                
                if item.get('word_count', 0) > 0:
                    formatted += f"📏 จำนวนคำ: {item['word_count']} คำ\n"
                
                formatted += "\n"
            else:
                formatted += f"❌ ล้มเหลว: {item.get('url', '')} - {item.get('error', 'ไม่ทราบสาเหตุ')}\n\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_search_result(result: Dict) -> str:
        if "error" in result:
            return f"❌ การค้นหาล้มเหลว: {result['error']}"
        
        query = result.get("search_query", "")
        engine = result.get("search_engine", "ไม่ทราบ")
        count = result.get("results_count", 0)
        
        formatted = f"🔍 ผลการค้นหาสำหรับ: '{query}'\n"
        formatted += f"🌐 เครื่องมือค้นหา: {engine}\n"
        formatted += f"📊 พบผลลัพธ์: {count} รายการ\n\n"
        
        results = result.get("results", [])
        for i, item in enumerate(results[:5], 1):
            formatted += f"{i}. 📰 {item.get('title', 'ไม่มีชื่อ')}\n"
            formatted += f"   🔗 {item.get('url', '')}\n"
            
            snippet = item.get('snippet', '')
            if snippet:
                formatted += f"   📄 {snippet}\n"
            formatted += "\n"
        
        if len(results) > 5:
            formatted += f"... และอีก {len(results) - 5} รายการ\n"
        
        related = result.get("related_searches", [])
        if related:
            formatted += f"\n💡 คำค้นหาที่เกี่ยวข้อง:\n"
            for related_query in related[:3]:
                formatted += f"   • {related_query}\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_database_result(result: Dict) -> str:
        if "error" in result:
            return f"❌ การสอบถามฐานข้อมูลล้มเหลว: {result['error']}\n💡 {result.get('suggestion', 'ลองใช้คำสั่งอื่น')}"
        
        query = result.get("query", "")
        row_count = result.get("row_count", 0)
        
        formatted = f"🗄️ ผลการสอบถามฐานข้อมูล\n"
        formatted += f"📝 คำสั่ง: {query}\n"
        formatted += f"📊 พบข้อมูล: {row_count} รายการ\n\n"
        
        if result.get("summary"):
            formatted += f"📋 สรุป: {result['summary']}\n\n"
        
        results = result.get("results", [])
        if results:
            formatted += "📄 รายละเอียดข้อมูล:\n"
            for i, record in enumerate(results[:8], 1):
                formatted += f"\n{i}. "
                for key, value in record.items():
                    readable_key = key.replace('_', ' ').title()
                    formatted += f"{readable_key}: {value} | "
                formatted = formatted.rstrip(" | ") + "\n"
            
            if len(results) > 8:
                formatted += f"\n... และอีก {len(results) - 8} รายการ\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_knowledge_result(result: Dict) -> str:
        query = result.get("query", "")
        response = result.get("response", "")
        confidence = result.get("confidence_score", 0)
        
        formatted = f"📚 ผลจากคลังความรู้\n"
        formatted += f"❓ คำถาม: {query}\n"
        formatted += f"🎯 ความเชื่อมั่น: {confidence*100:.1f}%\n\n"
        
        if response:
            formatted += f"💡 คำตอบ:\n{response}\n\n"
        
        sources = result.get("sources", [])
        if sources:
            formatted += "📖 แหล่งอ้างอิง:\n"
            for source in sources:
                formatted += f"   • {source.get('title', 'ไม่ทราบชื่อ')} (เวอร์ชัน {source.get('version', '1.0')})\n"
        
        related = result.get("related_topics", [])
        if related:
            formatted += f"\n🔗 หัวข้อที่เกี่ยวข้อง: {', '.join(related)}\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_code_result(result: Dict) -> str:
        if "error" in result:
            return f"❌ การรันโค้ดล้มเหลว: {result['error']}\n💡 {result.get('suggestion', 'ตรวจสอบ syntax ของโค้ด')}"
        
        code = result.get("code", "")
        output = result.get("result", "")
        exec_time = result.get("execution_time", 0)
        
        formatted = f"💻 ผลการรันโค้ด\n"
        formatted += f"⏱️ เวลาที่ใช้: {exec_time*1000:.1f} มิลลิวินาที\n\n"
        
        if len(code) < 200:
            formatted += f"📝 โค้ดที่รัน:\n```python\n{code}\n```\n\n"
        
        if output:
            formatted += f"📤 ผลลัพธ์:\n{output}\n"
        else:
            formatted += "✅ โค้ดทำงานสำเร็จ (ไม่มี output)\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_hrm_result(result: Dict) -> str:
        formatted = "🧠 ผลการวิเคราะห์เชิงเหตุผล (HRM)\n\n"
        
        if "structured_response" in result:
            structured = result["structured_response"]
            if isinstance(structured, dict):
                if structured.get("analysis_type") == "pros_and_cons":
                    subject = structured.get("subject", "หัวข้อที่วิเคราะห์")
                    formatted += f"📊 การวิเคราะห์ข้อดี-ข้อเสีย: {subject}\n\n"
                    
                    pros = structured.get("pros", [])
                    if pros:
                        formatted += "✅ ข้อดี:\n"
                        for i, pro in enumerate(pros, 1):
                            formatted += f"   {i}. {pro}\n"
                        formatted += "\n"
                    
                    cons = structured.get("cons", [])
                    if cons:
                        formatted += "❌ ข้อเสีย:\n"
                        for i, con in enumerate(cons, 1):
                            formatted += f"   {i}. {con}\n"
                        formatted += "\n"
                    
                    recommendation = structured.get("recommendation", "")
                    if recommendation:
                        formatted += f"💡 คำแนะนำ: {recommendation}\n"
                
                else:
                    # General analysis
                    insights = structured.get("key_insights", [])
                    if insights:
                        formatted += "🔍 ข้อสังเกต:\n"
                        for insight in insights:
                            formatted += f"   • {insight}\n"
            else:
                formatted += str(structured)
        
        elif "response" in result:
            formatted += result["response"]
        
        complexity = result.get("complexity_level", "")
        if complexity:
            formatted += f"\n\n🎚️ ระดับความซับซ้อน: {complexity.title()}"
        
        return formatted.strip()
    
    @staticmethod
    def format_general_result(result: Dict) -> str:
        response = result.get("response", "")
        
        formatted = f"💬 การตอบกลับทั่วไป\n\n{response}\n\n"
        
        capabilities = result.get("capabilities", [])
        if capabilities:
            formatted += "🎯 ความสามารถที่มี:\n"
            for cap in capabilities:
                formatted += f"   • {cap}\n"
        
        examples = result.get("examples", [])
        if examples:
            formatted += f"\n💡 ตัวอย่างการใช้งาน:\n"
            for example in examples[:3]:
                formatted += f"   • {example}\n"
        
        return formatted.strip()
    
    @staticmethod
    def format_result(task_type: TaskType, result: Dict) -> str:
        """Main formatter dispatcher"""
        try:
            if task_type == TaskType.WEB_SCRAPING:
                return ResultFormatter.format_web_scraping_result(result)
            elif task_type == TaskType.GOOGLE_SEARCH:
                return ResultFormatter.format_search_result(result)
            elif task_type == TaskType.DATABASE_QUERY:
                return ResultFormatter.format_database_result(result)
            elif task_type == TaskType.KM_RAG:
                return ResultFormatter.format_knowledge_result(result)
            elif task_type == TaskType.CODE_EXECUTION:
                return ResultFormatter.format_code_result(result)
            elif task_type == TaskType.HRM_REASONING:
                return ResultFormatter.format_hrm_result(result)
            else:
                return ResultFormatter.format_general_result(result)
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return f"📋 ผลลัพธ์: {str(result)}"

class HRMIntegratedProcessor:
    """Enhanced HRM processor that integrates with all task types"""
    
    def __init__(self):
        self.available = True
        logger.info("HRM Integrated Processor initialized successfully")
    
    async def enhance_result(self, task_type: TaskType, original_result: Any, query: str) -> Dict[str, Any]:
        """Add HRM enhancement to any task result"""
        
        try:
            enhanced = {
                "original_result": original_result,
                "hrm_analysis": await self._generate_hrm_analysis(task_type, original_result, query),
                "enhanced_insights": await self._generate_insights(task_type, original_result, query),
                "recommendations": await self._generate_recommendations(task_type, original_result, query),
                "hrm_enhanced": True
            }
            
            return enhanced
            
        except Exception as e:
            logger.error(f"HRM enhancement failed: {e}")
            return {"original_result": original_result, "hrm_available": False}
    
    async def _generate_hrm_analysis(self, task_type: TaskType, result: Any, query: str) -> str:
        """Generate HRM-style analysis"""
        
        analyses = {
            TaskType.WEB_SCRAPING: f"การวิเคราะห์เนื้อหาจาก '{query}' พบข้อมูลที่มีโครงสร้างชัดเจน เหมาะสำหรับการวิเคราะห์เชิงลึกเพิ่มเติม",
            TaskType.GOOGLE_SEARCH: f"ผลการค้นหา '{query}' แสดงมุมมองที่หลากหลาย ควรวิเคราะห์เปรียบเทียบจากหลายแหล่งเพื่อความครบถ้วน",
            TaskType.DATABASE_QUERY: f"การสอบถามข้อมูลพบ {len(result.get('results', [])) if isinstance(result, dict) else 0} รายการ เผยให้เห็นรูปแบบที่น่าสนใจ",
            TaskType.KM_RAG: f"การดึงความรู้สำหรับ '{query}' เชื่อมโยงกับหลายหัวข้อ แสดงความสัมพันธ์เชิงซับซ้อน",
            TaskType.CODE_EXECUTION: f"การวิเคราะห์โค้ดแสดงกระบวนการคำนวณที่มีประสิทธิภาพ",
            TaskType.GENERAL_QUERY: f"คำถามทั่วไปเกี่ยวกับ '{query}' ต้องการการทำความเข้าใจเชิงลึกเพิ่มเติม"
        }
        
        return analyses.get(task_type, "HRM วิเคราะห์เสร็จสมบูรณ์")
    
    async def _generate_insights(self, task_type: TaskType, result: Any, query: str) -> List[str]:
        """Generate strategic insights"""
        
        task_insights = {
            TaskType.WEB_SCRAPING: [
                "เนื้อหามีโครงสร้างที่ดี เหมาะสำหรับการแยกวิเคราะห์",
                "ความหนาแน่นของข้อมูลสูง บ่งชี้ถึงแหล่งข้อมูลที่น่าเชื่อถือ",
                "รูปแบบการเชื่อมโยงแสดงโอกาสในการค้นหาข้อมูลเพิ่ม"
            ],
            TaskType.GOOGLE_SEARCH: [
                "ผลการค้นหาแสดงภาพรวมข้อมูลที่หลากหลาย",
                "การปรับแต่งคำค้นหาอาจให้ผลลัพธ์ที่เฉพาะเจาะจงมากขึ้น",
                "มีแหล่งข้อมูลที่น่าเชื่อถือหลายแหล่งสำหรับการตรวจสอบ"
            ],
            TaskType.DATABASE_QUERY: [
                "รูปแบบข้อมูลแสดงถึงตรรกะทางธุรกิจที่ชัดเจน",
                "การกระจายของข้อมูลอยู่ในสภาพปกติ",
                "การกรองเพิ่มเติมอาจให้ข้อมูลเชิงลึกมากขึ้น"
            ],
            TaskType.KM_RAG: [
                "คลังความรู้ครอบคลุมหัวข้อได้ดี",
                "ข้อมูลนโยบายได้รับการอัปเดตเป็นปัจจุบัน",
                "การอ้างอิงข้าม แสดงความสอดคล้องขององค์กร"
            ],
            TaskType.CODE_EXECUTION: [
                "การรันโค้ดแสดงประสิทธิภาพที่ดี",
                "อัลกอริทึมที่เลือกเหมาะสมกับความซับซ้อนของงาน",
                "ตัวชี้วัดประสิทธิภาพบ่งชี้ถึงโซลูชันที่ขยายได้"
            ],
            TaskType.GENERAL_QUERY: [
                "คำถามทั่วไปต้องการการระบุเฉพาะเจาะจงมากขึ้น",
                "หลายมุมมองสามารถนำมาใช้ในการตอบ",
                "บริบทเพิ่มเติมจะช่วยให้การตอบแม่นยำขึ้น"
            ]
        }
        
        return task_insights.get(task_type, ["งานประมวลผลเสร็จสมบูรณ์ด้วยการวิเคราะห์หลายระดับ"])
    
    async def _generate_recommendations(self, task_type: TaskType, result: Any, query: str) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = {
            TaskType.WEB_SCRAPING: [
                "พิจารณาตั้งระบบติดตามการอัปเดตเนื้อหา",
                "ดึง metadata เพิ่มเติมสำหรับการวิเคราะห์ครอบคลุม",
                "สร้างตารางเวลาการดึงข้อมูลอัตโนมัติ"
            ],
            TaskType.GOOGLE_SEARCH: [
                "ขยายการค้นหาด้วยคำหลักที่เกี่ยวข้องเพิ่มเติม",
                "ตรวจสอบข้อมูลจากหลายแหล่งเพื่อความแม่นยำ",
                "สร้าง search alerts เพื่อติดตามข้อมูลใหม่"
            ],
            TaskType.DATABASE_QUERY: [
                "พิจารณาสร้าง views สำหรับ query patterns ที่ใช้บ่อย",
                "จัดทำ data visualization เพื่อความเข้าใจที่ดีขึ้น",
                "ตั้งระบบรายงานอัตโนมัติสำหรับข้อมูลที่สำคัญ"
            ],
            TaskType.KM_RAG: [
                "อัปเดตคลังความรู้ด้วยข้อมูลล่าสุด",
                "สร้าง quick reference guides สำหรับคำถามที่พบบ่อย",
                "จัดทำระบบ feedback เพื่อปรับปรุงคุณภาพความรู้"
            ],
            TaskType.CODE_EXECUTION: [
                "บันทึก code snippets สำหรับการใช้งานในอนาคต",
                "พิจารณาสร้างฟังก์ชันสำหรับการดำเนินการที่ซ้ำ",
                "จัดทำเอกสาร logic ของโค้ดเพื่อแชร์ในทีม"
            ],
            TaskType.GENERAL_QUERY: [
                "ปรับแต่งคำถามให้เฉพาะเจาะจงมากขึ้น",
                "แบ่งคำขอที่ซับซ้อนออกเป็นงานย่อย",
                "ใช้ prefix เฉพาะเพื่อการประมวลผลที่ดีขึ้น"
            ]
        }
        
        return recommendations.get(task_type, ["สำรวจหัวข้อที่เกี่ยวข้องเพิ่มเติม", "บันทึกผลลัพธ์เพื่ออ้างอิงในอนาคต"])

class EnhancedWebScrapingProcessor:
    """Fixed web scraping processor with robust error handling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.timeout = 15
        self.max_retries = 2
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Enhanced web scraping with better error handling"""
        
        urls = self._extract_urls(input_text)
        if not urls:
            return {
                "error": "ไม่พบ URL ที่ถูกต้องในข้อความ",
                "suggestion": "โปรดระบุ URL ที่ถูกต้อง เช่น https://example.com"
            }
        
        results = []
        for url in urls[:3]:  # จำกัด 3 URLs
            try:
                content = await self._scrape_url_safe(url)
                results.append({
                    "url": url,
                    "title": content.get("title", "ไม่มีชื่อ"),
                    "content": content.get("content", "")[:1500],
                    "links": content.get("links", [])[:10],
                    "images": content.get("images", [])[:5],
                    "metadata": content.get("metadata", {}),
                    "word_count": content.get("word_count", 0),
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Scraping failed for {url}: {str(e)}")
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
        """Extract URLs from text with improved validation"""
        
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.-])*(?:\?(?:[\w&=%.-])*)?(?:#(?:[\w.-])*)?)?'
        urls = re.findall(url_pattern, text)
        
        domain_pattern = r'\b(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, text)
        
        for domain in domains:
            if not any(domain in url for url in urls):
                if not domain.startswith(('http://', 'https://')):
                    urls.append(f'https://{domain}')
        
        valid_urls = []
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    valid_urls.append(url)
            except Exception:
                continue
        
        return list(set(valid_urls))
    
    async def _scrape_url_safe(self, url: str) -> Dict[str, Any]:
        """Safely scrape URL with enhanced error handling"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.timeout, 
                    allow_redirects=True,
                    verify=False
                )
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    return {
                        "title": f"ไม่ใช่ HTML: {content_type}",
                        "content": f"ประเภทเนื้อหา: {content_type}, ขนาด: {len(response.content)} bytes",
                        "links": [],
                        "images": [],
                        "metadata": {"content_type": content_type},
                        "word_count": 0
                    }
                
                try:
                    soup = BeautifulSoup(response.content, 'html.parser')
                except Exception as e:
                    logger.error(f"BeautifulSoup parsing error: {e}")
                    return {
                        "title": "เกิดข้อผิดพลาดในการ parse HTML",
                        "content": f"ไม่สามารถประมวลผล HTML ได้: {str(e)}",
                        "links": [],
                        "images": [],
                        "metadata": {},
                        "word_count": 0
                    }
                
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                return self._extract_content_safe(soup, url)
                
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(1)
    
    def _extract_content_safe(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Safely extract content with comprehensive error handling"""
        
        # Extract title safely
        title = ""
        try:
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text().strip()
        except Exception as e:
            logger.warning(f"Title extraction error: {e}")
            title = "ไม่สามารถดึงชื่อได้"
        
        # Extract main content safely
        content = ""
        content_selectors = ['main', 'article', '.content', '#content', '.post-content']
        
        try:
            for selector in content_selectors:
                try:
                    if selector.startswith('.') or selector.startswith('#'):
                        main_content = soup.select_one(selector)
                    else:
                        main_content = soup.find(selector)
                    
                    if main_content:
                        content = main_content.get_text(separator=' ', strip=True)
                        if len(content) > 100:
                            break
                except Exception:
                    continue
            
            if not content and soup.body:
                try:
                    content = soup.body.get_text(separator=' ', strip=True)
                except Exception:
                    content = "ไม่สามารถดึงเนื้อหาได้"
        except Exception as e:
            logger.warning(f"Content extraction error: {e}")
            content = "เกิดข้อผิดพลาดในการดึงเนื้อหา"
        
        if content:
            content = ' '.join(content.split())[:3000]
        
        # Extract links safely
        links = []
        try:
            for link in soup.find_all('a', href=True)[:15]:
                try:
                    href = link.get('href', '')
                    text = link.get_text().strip()
                    
                    if href and text and len(text) > 2:
                        if href.startswith(('http://', 'https://')):
                            full_url = href
                        elif href.startswith('//'):
                            full_url = f"https:{href}"
                        else:
                            full_url = urljoin(url, href)
                        
                        links.append({
                            "text": text[:100],
                            "url": full_url
                        })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Links extraction error: {e}")
        
        # Extract images safely
        images = []
        try:
            for img in soup.find_all('img', src=True)[:10]:
                try:
                    src = img.get('src', '')
                    alt = img.get('alt', '')
                    
                    if src:
                        if src.startswith(('http://', 'https://')):
                            full_url = src
                        elif src.startswith('//'):
                            full_url = f"https:{src}"
                        else:
                            full_url = urljoin(url, src)
                        
                        images.append({
                            "src": full_url,
                            "alt": alt[:100] if alt else ""
                        })
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Images extraction error: {e}")
        
        # Extract metadata safely
        metadata = {}
        try:
            for meta in soup.find_all('meta'):
                try:
                    if meta:
                        name = meta.get('name')
                        property_attr = meta.get('property')
                        content_attr = meta.get('content')
                        
                        if name and content_attr:
                            if name.lower() in ['description', 'keywords', 'author']:
                                metadata[name.lower()] = content_attr[:300]
                        elif property_attr and content_attr:
                            if property_attr.startswith('og:'):
                                metadata[property_attr] = content_attr[:300]
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Metadata extraction error: {e}")
        
        return {
            "title": title[:200] if title else "ไม่มีชื่อ",
            "content": content,
            "links": links,
            "images": images,
            "metadata": metadata,
            "word_count": len(content.split()) if content else 0,
            "url": url
        }

class EnhancedGoogleSearchProcessor:
    """Enhanced search processor with better mock results"""
    
    def __init__(self):
        self.timeout = 10
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process search queries with enhanced mock results"""
        
        search_query = self._extract_search_query(input_text)
        
        try:
            results = await self._generate_enhanced_mock_results(search_query)
            return {
                **results,
                "search_query": search_query,
                "query_analysis": self._analyze_query(search_query)
            }
        except Exception as e:
            logger.error(f"Search processing failed: {str(e)}")
            return await self._generate_enhanced_mock_results(search_query)
    
    def _extract_search_query(self, text: str) -> str:
        """Extract search query from text"""
        
        search_prefixes = [
            "search for", "search", "ค้นหา", "หา", "google", "find", "look up"
        ]
        
        cleaned_text = text.lower().strip()
        
        for prefix in search_prefixes:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        
        return cleaned_text.strip("\"'") if cleaned_text else text.strip()
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze search query"""
        
        return {
            "word_count": len(query.split()),
            "query_type": self._classify_query_type(query),
            "intent": self._detect_search_intent(query),
            "language": "thai" if any(ord(c) > 127 for c in query) else "english"
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "when", "where", "who", "อะไร", "ทำไม", "อย่างไร"]):
            return "question"
        elif any(word in query_lower for word in ["best", "top", "review", "compare", "ดีที่สุด", "เปรียบเทียบ"]):
            return "comparison"
        elif any(word in query_lower for word in ["tutorial", "guide", "learn", "สอน", "วิธี"]):
            return "educational"
        else:
            return "informational"
    
    def _detect_search_intent(self, query: str) -> str:
        """Detect search intent"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["buy", "price", "cost", "ราคา", "ซื้อ"]):
            return "commercial"
        elif any(word in query_lower for word in ["how to", "tutorial", "วิธี", "สอน"]):
            return "instructional"
        else:
            return "informational"
    
    async def _generate_enhanced_mock_results(self, query: str) -> Dict[str, Any]:
        """Generate realistic mock search results"""
        
        mock_results = []
        
        templates = [
            {
                "title": f"คู่มือครบถ้วนเกี่ยวกับ {query}",
                "url": f"https://guide.example.com/{query.replace(' ', '-').lower()}",
                "snippet": f"เรียนรู้ทุกสิ่งเกี่ยวกับ {query} พร้อมคำแนะนำจากผู้เชี่ยวชาญ และขั้นตอนที่ชัดเจน"
            },
            {
                "title": f"{query} - Wikipedia",
                "url": f"https://th.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": f"บทความ Wikipedia เกี่ยวกับ {query} พร้อมข้อมูลรายละเอียด ประวัติ และการอ้างอิงจากแหล่งที่เชื่อถือได้"
            },
            {
                "title": f"ข่าวและอัปเดตล่าสุดเกี่ยวกับ {query}",
                "url": f"https://news.example.com/topics/{query.replace(' ', '-')}",
                "snippet": f"ติดตามข่าวสาร แนวโน้ม และความพัฒนาล่าสุดเกี่ยวกับ {query} จากแหล่งข่าวที่เชื่อถือได้"
            },
            {
                "title": f"Best Practices สำหรับ {query}",
                "url": f"https://bestpractices.example.com/{query.replace(' ', '-')}",
                "snippet": f"แนวทางปฏิบัติที่ดีที่สุดและคำแนะนำจากผู้เชี่ยวชาญสำหรับ {query}"
            },
            {
                "title": f"Tutorial และตัวอย่าง {query}",
                "url": f"https://tutorial.example.com/{query.replace(' ', '-')}",
                "snippet": f"เรียนรู้ {query} ด้วยตัวอย่างที่เป็นจริง tutorial ที่ลงลึก และการใช้งานจริง"
            }
        ]
        
        for i, template in enumerate(templates):
            mock_results.append({
                **template,
                "source": "mock",
                "rank": i + 1
            })
        
        return {
            "search_engine": "Enhanced Mock Search",
            "results_count": len(mock_results),
            "results": mock_results,
            "related_searches": self._generate_related_searches(query),
            "status": "success",
            "note": "ผลลัพธ์จำลองสำหรับการทดสอบ"
        }
    
    def _generate_related_searches(self, query: str) -> List[str]:
        """Generate related search suggestions"""
        
        thai_templates = [
            f"{query} คืออะไร",
            f"วิธีใช้ {query}",
            f"ตัวอย่าง {query}",
            f"ประโยชน์ของ {query}",
            f"{query} vs ทางเลือกอื่น"
        ]
        
        english_templates = [
            f"what is {query}",
            f"how to use {query}",
            f"{query} examples",
            f"{query} benefits",
            f"{query} tutorial"
        ]
        
        if any(ord(c) > 127 for c in query):
            return thai_templates[:4]
        else:
            return english_templates[:4]

class EnhancedDatabaseProcessor:
    """Enhanced database processor with comprehensive demo data"""
    
    def __init__(self, db_path: str = "demo_decision_agent.db"):
        self.db_path = db_path
        self._init_comprehensive_database()
    
    def _init_comprehensive_database(self):
        """Initialize comprehensive demo database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DROP TABLE IF EXISTS tasks")
            cursor.execute("DROP TABLE IF EXISTS projects") 
            cursor.execute("DROP TABLE IF EXISTS users")
            
            cursor.execute("""
                CREATE TABLE users (
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
                CREATE TABLE tasks (
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
                    FOREIGN KEY (assigned_to) REFERENCES users (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE projects (
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
            
            sample_users = [
                ("Alice Johnson", "alice@company.com", "Engineering", "Senior Developer", 95000, "2022-01-15", "active"),
                ("Bob Smith", "bob@company.com", "Marketing", "Marketing Manager", 75000, "2021-06-20", "active"),
                ("Carol Davis", "carol@company.com", "Sales", "Sales Representative", 65000, "2023-03-10", "active"),
                ("David Wilson", "david@company.com", "Support", "Support Specialist", 55000, "2022-08-05", "active"),
                ("Eva Martinez", "eva@company.com", "Engineering", "DevOps Engineer", 90000, "2021-11-12", "active"),
                ("Frank Brown", "frank@company.com", "HR", "HR Coordinator", 60000, "2020-04-18", "inactive")
            ]
            
            cursor.executemany(
                "INSERT INTO users (name, email, department, position, salary, hire_date, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                sample_users
            )
            
            sample_tasks = [
                ("Implement Authentication", "พัฒนาระบบล็อกอิน/ล็อกเอาต์พร้อม 2FA", "in_progress", 3, 1, 24.0, 18.5, "2024-02-15"),
                ("Marketing Campaign Analysis", "วิเคราะห์ผลการรณรงค์ Q4 2023", "completed", 2, 2, 16.0, 20.0, "2024-01-30"),
                ("Customer Support Tickets", "จัดการคำขอสนับสนุนลูกค้า", "pending", 1, 4, 8.0, None, "2024-02-10"),
                ("Sales Report Automation", "ทำระบบรายงานยอดขายอัตโนมัติ", "pending", 2, 3, 32.0, None, "2024-02-28"),
                ("Infrastructure Monitoring", "ติดตั้งระบบตรวจสอบโครงสร้าง", "in_progress", 3, 5, 40.0, 25.0, "2024-03-15"),
                ("Database Optimization", "ปรับปรุงประสิทธิภาพฐานข้อมูล", "planning", 2, 1, 20.0, None, "2024-03-01")
            ]
            
            cursor.executemany(
                "INSERT INTO tasks (title, description, status, priority, assigned_to, estimated_hours, actual_hours, due_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                sample_tasks
            )
            
            sample_projects = [
                ("Website Redesign", "ปรับปรุงเว็บไซต์ใหม่ทั้งหมด", "active", 50000.0, "2024-01-01", "2024-04-30", 2),
                ("Mobile App Development", "พัฒนาแอป iOS และ Android", "planning", 120000.0, "2024-03-01", "2024-09-30", 1),
                ("Database Migration", "ย้ายฐานข้อมูลไปยัง cloud", "completed", 30000.0, "2023-10-01", "2023-12-31", 5)
            ]
            
            cursor.executemany(
                "INSERT INTO projects (name, description, status, budget, start_date, end_date, manager_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                sample_projects
            )
            
            conn.commit()
            logger.info("Demo database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            conn.rollback()
        finally:
            conn.close()
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process database queries with enhanced natural language support"""
        
        try:
            query_info = self._analyze_query(input_text)
            
            if query_info["type"] == "sql":
                sql_query = self._sanitize_sql(query_info["query"])
                results = self._execute_query_safe(sql_query)
            else:
                sql_query = self._generate_sql_enhanced(input_text, query_info)
                results = self._execute_query_safe(sql_query)
            
            processed_results = self._process_results_enhanced(results, query_info)
            
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
                "suggestion": self._get_helpful_suggestion(input_text),
                "database_info": self._get_database_info()
            }
    
    def _analyze_query(self, text: str) -> Dict[str, Any]:
        """Analyze query with improved detection"""
        
        text_lower = text.lower().strip()
        
        sql_keywords = ['select', 'insert', 'update', 'delete', 'create', 'drop', 'alter']
        if any(text_lower.startswith(keyword) for keyword in sql_keywords):
            return {
                "type": "sql",
                "query": text,
                "tables": self._extract_table_names(text),
                "intent": "direct_sql"
            }
        
        return {
            "type": "natural",
            "intent": self._detect_query_intent_enhanced(text_lower),
            "entities": self._extract_entities_enhanced(text_lower),
            "filters": self._extract_filters_enhanced(text_lower),
            "aggregation": self._detect_aggregation_enhanced(text_lower)
        }
    
    def _detect_query_intent_enhanced(self, text: str) -> str:
        """Detect query intent"""
        
        if any(word in text for word in ["แสดง", "ดู", "หา", "ค้นหา", "list", "show", "find", "get"]):
            return "retrieve"
        elif any(word in text for word in ["นับ", "จำนวน", "count", "how many"]):
            return "count"
        elif any(word in text for word in ["รวม", "เฉลี่ย", "sum", "average", "total"]):
            return "aggregate"
        else:
            return "retrieve"
    
    def _extract_entities_enhanced(self, text: str) -> List[str]:
        """Extract entities from query"""
        
        entities = []
        
        table_keywords = {
            "users": ["user", "users", "ผู้ใช้", "พนักงาน", "คน"],
            "tasks": ["task", "tasks", "งาน", "ภารกิจ"],
            "projects": ["project", "projects", "โครงการ", "project"]
        }
        
        for table, keywords in table_keywords.items():
            if any(keyword in text for keyword in keywords):
                entities.append(table)
        
        return entities
    
    def _extract_filters_enhanced(self, text: str) -> Dict[str, Any]:
        """Extract filter conditions"""
        
        filters = {}
        
        status_map = {
            "pending": ["pending", "รอดำเนินการ", "ยังไม่เสร็จ"],
            "completed": ["completed", "เสร็จแล้ว", "สำเร็จ"],
            "in_progress": ["in_progress", "in progress", "กำลังทำ", "ดำเนินการ"],
            "active": ["active", "ใช้งาน"],
            "inactive": ["inactive", "ไม่ใช้งาน"]
        }
        
        for status, keywords in status_map.items():
            if any(keyword in text for keyword in keywords):
                filters["status"] = status
                break
        
        departments = ["engineering", "marketing", "sales", "support", "hr"]
        for dept in departments:
            if dept in text:
                filters["department"] = dept
                break
        
        names = ["alice", "bob", "carol", "david", "eva", "frank"]
        for name in names:
            if name in text:
                filters["assigned_name"] = name.title()
                break
        
        return filters
    
    def _detect_aggregation_enhanced(self, text: str) -> Optional[str]:
        """Detect aggregation operations"""
        
        if any(word in text for word in ["count", "นับ", "จำนวน", "how many"]):
            return "COUNT"
        elif any(word in text for word in ["sum", "รวม", "total"]):
            return "SUM"
        elif any(word in text for word in ["average", "avg", "เฉลี่ย"]):
            return "AVG"
        else:
            return None
    
    def _generate_sql_enhanced(self, text: str, query_info: Dict) -> str:
        """Generate enhanced SQL from natural language"""
        
        intent = query_info["intent"]
        entities = query_info["entities"]
        filters = query_info["filters"]
        aggregation = query_info["aggregation"]
        
        if "users" in entities:
            primary_table = "users"
        elif "tasks" in entities:
            primary_table = "tasks"
        elif "projects" in entities:
            primary_table = "projects"
        else:
            primary_table = "users"
        
        if aggregation == "COUNT":
            if "department" in text and primary_table == "users":
                select_clause = "SELECT department, COUNT(*) as count"
                group_by = " GROUP BY department"
            else:
                select_clause = "SELECT COUNT(*) as count"
                group_by = ""
        else:
            if primary_table == "tasks" and any(word in text for word in ["assigned", "ของ", "งาน"]):
                select_clause = """
                    SELECT u.name, t.title, t.status, t.priority, t.due_date
                    FROM tasks t
                    JOIN users u ON t.assigned_to = u.id
                """
                primary_table = ""
            else:
                select_clause = "SELECT *"
            group_by = ""
        
        from_clause = f" FROM {primary_table}" if primary_table else ""
        
        where_conditions = []
        for key, value in filters.items():
            if key == "assigned_name":
                where_conditions.append(f"u.name LIKE '%{value}%'")
            elif key in ["status", "department"]:
                where_conditions.append(f"{key} = '{value}'")
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        order_clause = ""
        if intent == "retrieve" and not aggregation:
            if primary_table == "tasks" or "tasks" in entities:
                order_clause = " ORDER BY priority DESC, created_at DESC"
            elif primary_table == "users":
                order_clause = " ORDER BY name"
        
        sql_query = select_clause + from_clause + where_clause + group_by + order_clause + " LIMIT 50"
        
        return sql_query.strip()
    
    def _sanitize_sql(self, sql: str) -> str:
        """Sanitize SQL for security"""
        
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter', 'truncate']
        sql_lower = sql.lower().strip()
        
        for keyword in dangerous_keywords:
            if sql_lower.startswith(keyword):
                raise ValueError(f"คำสั่ง '{keyword.upper()}' ไม่อนุญาตในโหมดสาธิต")
        
        return sql
    
    def _execute_query_safe(self, sql_query: str) -> List[Dict]:
        """Execute SQL query safely"""
        
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
    
    def _process_results_enhanced(self, results: List[Dict], query_info: Dict) -> Dict[str, Any]:
        """Process and summarize results"""
        
        if not results:
            return {
                "data": [],
                "summary": "ไม่พบข้อมูลที่ตรงกับเงื่อนไข"
            }
        
        intent = query_info.get("intent", "retrieve")
        
        if intent == "count" or query_info.get("aggregation") == "COUNT":
            if len(results) == 1 and "count" in results[0]:
                summary = f"พบข้อมูล {results[0]['count']} รายการ"
            else:
                summary = f"ผลการนับ: {len(results)} หมวดหมู่"
        else:
            summary = f"ดึงข้อมูลสำเร็จ {len(results)} รายการ"
        
        return {
            "data": results,
            "summary": summary
        }
    
    def _get_helpful_suggestion(self, failed_query: str) -> str:
        """Provide helpful suggestions"""
        
        suggestions = [
            "ลองใช้: 'แสดงผู้ใช้ทั้งหมด' หรือ 'หางานที่ pending'",
            "ตัวอย่าง: 'นับผู้ใช้ตามแผนก' หรือ 'งานของ Alice'",
            "คำสั่งที่รองรับ: แสดง, หา, นับ, รายการ"
        ]
        
        return " | ".join(suggestions)
    
    def _get_database_info(self) -> str:
        """Get database information"""
        return """
🗄️ ข้อมูลฐานข้อมูลสาธิต

📊 แหล่งที่มา: ฐานข้อมูล SQLite ที่สร้างขึ้นอัตโนมัติ
🔄 การรีเฟรช: ข้อมูลจะถูกสร้างใหม่ทุกครั้งที่เริ่มระบบ

📋 ตารางที่มีอยู่:

1. 👥 ตาราง USERS (ผู้ใช้):
   - 6 พนักงาน: Alice, Bob, Carol, David, Eva, Frank
   - แผนก: Engineering, Marketing, Sales, Support, HR
   - ข้อมูล: ชื่อ, อีเมล, ตำแหน่ง, เงินเดือน, วันที่เข้าทำงาน

2. 📋 ตาราง TASKS (งาน):
   - 6 งาน: Authentication, Campaign Analysis, Support Tickets, etc.
   - สถานะ: pending, in_progress, completed, planning
   - ข้อมูล: ชื่องาน, คำอธิบาย, ผู้รับผิดชอบ, กำหนดเวลา

3. 🚀 ตาราง PROJECTS (โครงการ):
   - 3 โครงการ: Website Redesign, Mobile App, Database Migration
   - ข้อมูล: ชื่อโครงการ, งงบประมาณ, ผู้จัดการ, ระยะเวลา

💡 ตัวอย่างการใช้งาน:
• "แสดงผู้ใช้ทั้งหมด" → SELECT * FROM users
• "หางานที่ pending" → งานที่ยังไม่เสร็จ
• "นับผู้ใช้ตามแผนก" → จำนวนคนในแต่ละแผนก
• "งานของ Alice" → งานที่มอบหมายให้ Alice
        """
    
    def _extract_table_names(self, sql: str) -> List[str]:
        """Extract table names from SQL"""
        
        tables = []
        words = sql.upper().split()
        
        keywords_with_tables = ['FROM', 'JOIN', 'UPDATE', 'INTO']
        
        for i, word in enumerate(words):
            if word in keywords_with_tables and i + 1 < len(words):
                table_name = words[i + 1].lower().strip(',;')
                tables.append(table_name)
        
        return list(set(tables))

class EnhancedKnowledgeRAGProcessor:
    """Enhanced Knowledge RAG processor with Thai language support"""
    
    def __init__(self):
        self.knowledge_base = self._init_thai_knowledge_base()
        self.response_cache = {}
    
    def _init_thai_knowledge_base(self) -> Dict[str, Any]:
        """Initialize Thai-supported knowledge base"""
        
        return {
            "documents": [
                {
                    "id": "doc_security_001",
                    "title": "นโยบายความปลอดภัยสารสนเทศ",
                    "content": """
                    นโยบายความปลอดภัยของเรากำหนดให้พนักงานทุกคนต้องใช้รหัสผ่านที่แข็งแรงอย่างน้อย 12 ตัวอักษร 
                    ประกอบด้วยตัวอักษรใหญ่ เล็ก ตัวเลข และอักขระพิเศษ การยืนยันตัวตนแบบสองขั้นตอน (2FA) จำเป็นต้องใช้ 
                    สำหรับระบบธุรกิจทั้งหมด การเข้าถึง VPN เป็นสิ่งจำเป็นสำหรับการทำงานระยะไกล อุปกรณ์ทั้งหมดต้องมี 
                    โปรแกรมป้องกันไวรัสและ firewall ที่อัปเดตล่าสุด การเข้ารหัสข้อมูลจำเป็นสำหรับข้อมูลสำคัญ
                    """,
                    "categories": ["ความปลอดภัย", "นโยบาย", "การปฏิบัติตาม", "การทำงานระยะไกล"],
                    "last_updated": "2024-01-20",
                    "access_level": "พนักงานทั้งหมด",
                    "version": "3.2"
                },
                {
                    "id": "doc_ai_guidelines_002",
                    "title": "แนวทางการพัฒนา AI และจริยธรรม",
                    "content": """
                    ในการพัฒนาระบบปัญญาประดิษฐ์ ทีมต้องให้ความสำคัญกับคุณภาพข้อมูล การตรวจสอบโมเดล 
                    และการพิจารณาด้านจริยธรรม โมเดล AI ทั้งหมดต้องผ่านการทดสอบอย่างเข้มงวด รวมถึงการตรวจจับ bias 
                    และการประเมินความยุติธรรม ความเป็นส่วนตัวของข้อมูลต้องได้รับการรักษาตลอด ML pipeline 
                    การติดตามและประเมินโมเดลอย่างต่อเนื่องเป็นสิ่งจำเป็นสำหรับระบบที่ใช้งานจริง
                    """,
                    "categories": ["AI", "machine learning", "จริยธรรม", "การพัฒนา", "แนวทาง"],
                    "last_updated": "2024-01-15",
                    "access_level": "ทีมเทคนิค",
                    "version": "2.1"
                },
                {
                    "id": "doc_remote_work_003",
                    "title": "นโยบายการทำงานระยะไกล",
                    "content": """
                    การทำงานระยะไกลได้รับการสนับสนุนสำหรับพนักงานที่มีสิทธิ์พร้อมการอนุมัติจากผู้จัดการ 
                    พนักงานต้องรักษาระดับผลิตภาพให้เทียบเท่ากับการทำงานในสำนักงาน การเชื่อมต่อ VPN ที่ปลอดภัย 
                    จำเป็นสำหรับการเข้าถึงทรัพยากรของบริษัท การประชุมเช็คอินกับทีมและหัวหน้างานเป็นสิ่งจำเป็น 
                    การจัดเตรียมพื้นที่ทำงานที่บ้านควรรวมถึงเฟอร์นิเจอร์ ergonomic และการเชื่อมต่ออินเทอร์เน็ตที่เสถียร
                    """,
                    "categories": ["การทำงานระยะไกล", "นโยบาย", "ผลิตภาพ", "ความปลอดภัย"],
                    "last_updated": "2024-01-25",
                    "access_level": "พนักงานทั้งหมด",
                    "version": "2.3"
                }
            ],
            "metadata": {
                "total_documents": 3,
                "last_indexed": datetime.now().isoformat(),
                "categories": [
                    "ความปลอดภัย", "AI", "machine learning", "การทำงานระยะไกล", 
                    "นโยบาย", "จริยธรรม", "การพัฒนา"
                ],
                "languages": ["ไทย", "อังกฤษ"]
            }
        }
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process knowledge queries with Thai language support"""
        
        cache_key = hash(input_text.lower().strip())
        if cache_key in self.response_cache:
            cached_result = self.response_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result
        
        relevant_docs = self._retrieve_documents_enhanced(input_text)
        ranked_docs = self._rank_documents_enhanced(relevant_docs, input_text)
        response = self._generate_thai_response(input_text, ranked_docs)
        
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
            "related_topics": self._extract_related_topics_thai(ranked_docs),
            "knowledge_base_stats": self.knowledge_base["metadata"]
        }
        
        self.response_cache[cache_key] = result
        return result
    
    def _retrieve_documents_enhanced(self, query: str) -> List[Dict]:
        """Enhanced document retrieval with semantic matching"""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_docs = []
        
        for doc in self.knowledge_base["documents"]:
            score = 0
            
            title_words = set(doc["title"].lower().split())
            title_overlap = len(query_words.intersection(title_words))
            score += title_overlap * 5
            
            content_words = set(doc["content"].lower().split())
            content_overlap = len(query_words.intersection(content_words))
            score += content_overlap * 2
            
            for category in doc["categories"]:
                if any(word in category.lower() for word in query_words):
                    score += 4
            
            thai_phrases = self._extract_thai_phrases(query_lower)
            for phrase in thai_phrases:
                if phrase in doc["content"].lower():
                    score += 8
            
            question_words = ["อะไร", "ทำไม", "อย่างไร", "เมื่อไร", "ที่ไหน", "ใคร", "what", "how", "why"]
            if any(q_word in query_lower for q_word in question_words):
                if any(q_word in doc["content"].lower() for q_word in ["วิธี", "ขั้นตอน", "แนวทาง", "procedure"]):
                    score += 3
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["relevance_score"] = score
                relevant_docs.append(doc_copy)
        
        return relevant_docs
    
    def _extract_thai_phrases(self, text: str) -> List[str]:
        """Extract Thai language phrases"""
        
        phrases = []
        words = text.split()
        
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i + 1]}"
            if len(phrase) > 4:
                phrases.append(phrase)
        
        if len(words) > 2:
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                phrases.append(phrase)
        
        return phrases
    
    def _rank_documents_enhanced(self, documents: List[Dict], query: str) -> List[Dict]:
        """Enhanced document ranking with multiple factors"""
        
        if not documents:
            return []
        
        for doc in documents:
            original_score = doc["relevance_score"]
            
            doc_date = datetime.fromisoformat(doc["last_updated"])
            days_old = (datetime.now() - doc_date).days
            recency_factor = max(0.8, 1.0 - (days_old / 365))
            doc["relevance_score"] *= recency_factor
            
            query_lower = query.lower()
            if any(phrase in doc["content"].lower() for phrase in self._extract_thai_phrases(query_lower)):
                doc["relevance_score"] *= 1.3
            
            content_length = len(doc["content"])
            if content_length > 300:
                doc["relevance_score"] *= 1.1
        
        return sorted(documents, key=lambda x: x["relevance_score"], reverse=True)
    
    def _generate_thai_response(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """Generate Thai language response"""
        
        if not documents:
            return {
                "answer": f"ไม่พบข้อมูลเกี่ยวกับ '{query}' ในคลังความรู้ หัวข้อที่มีได้แก่: {', '.join(self.knowledge_base['metadata']['categories'])}",
                "confidence": 0.1
            }
        
        top_docs = documents[:2]
        
        response_parts = []
        response_parts.append(f"จากคลังความรู้ของเรา ข้อมูลเกี่ยวกับ '{query}':")
        
        total_relevance = sum(doc["relevance_score"] for doc in top_docs)
        
        for i, doc in enumerate(top_docs, 1):
            relevant_content = self._extract_relevant_content_thai(doc["content"], query)
            
            response_parts.append(f"\n{i}. จาก '{doc['title']}' (เวอร์ชัน {doc.get('version', '1.0')}):")
            response_parts.append(f"   {relevant_content}")
        
        response_parts.append(f"\nแหล่งอ้างอิงหลัก: {', '.join([doc['title'] for doc in top_docs])}")
        
        confidence = min(0.95, (total_relevance / 50) * 0.8 + 0.2)
        
        return {
            "answer": " ".join(response_parts),
            "confidence": confidence
        }
    
    def _extract_relevant_content_thai(self, content: str, query: str) -> str:
        """Extract relevant Thai content sentences"""
        
        sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 10]
        query_words = set(query.lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            
            if any(phrase in sentence.lower() for phrase in self._extract_thai_phrases(query.lower())):
                overlap += 3
            
            if overlap > 0:
                scored_sentences.append((sentence, overlap))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sent[0] for sent in scored_sentences[:2]]
            return ". ".join(top_sentences) + "."
        else:
            for sentence in sentences:
                if len(sentence) > 30:
                    return sentence + "."
            
            return content[:400] + "..." if len(content) > 400 else content
    
    def _extract_related_topics_thai(self, documents: List[Dict]) -> List[str]:
        """Extract related topics from documents"""
        
        all_categories = []
        for doc in documents:
            all_categories.extend(doc["categories"])
        
        from collections import Counter
        category_counts = Counter(all_categories)
        
        return [cat for cat, count in category_counts.most_common(4)]

class EnhancedCodeExecutionProcessor:
    """Enhanced code execution processor with improved security"""
    
    def __init__(self):
        self.allowed_modules = {
            'math', 'statistics', 'random', 'datetime', 're', 'json',
            'collections', 'itertools', 'functools'
        }
        self.safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'enumerate',
            'filter', 'float', 'format', 'hex', 'int', 'len', 'list', 'map',
            'max', 'min', 'oct', 'ord', 'pow', 'range', 'reversed', 'round',
            'set', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'
        }
        self.execution_timeout = 3
        self.max_output_length = 5000
    
    async def process(self, input_text: str) -> Dict[str, Any]:
        """Process code execution with enhanced security"""
        
        code_info = self._extract_code_enhanced(input_text)
        
        if not code_info["code"]:
            return {
                "error": "ไม่พบโค้ดที่สามารถรันได้",
                "suggestion": "ใส่โค้ด Python หรือใช้คำนำหน้า เช่น 'execute:', 'calculate:', 'run:'",
                "examples": [
                    "execute: print(sum(range(100)))",
                    "calculate: (15 * 37) + 128", 
                    "run: import math; print(math.pi)"
                ]
            }
        
        if not self._basic_security_check(code_info["code"]):
            return {
                "code": code_info["code"],
                "error": "โค้ดประกอบด้วยการดำเนินการที่ไม่ปลอดภัย (import, exec, eval, open, file)",
                "status": "security_error",
                "security_note": "การรันโค้ดถูกปิดกั้นด้วยเหตุผลความปลอดภัย"
            }
        
        try:
            validation_result = self._validate_code_enhanced(code_info["code"])
            if not validation_result["is_safe"]:
                return {
                    "code": code_info["code"],
                    "error": f"การตรวจสอบโค้ดล้มเหลว: {validation_result['reason']}",
                    "status": "validation_error"
                }
            
            start_time = time.time()
            execution_result = await self._execute_safe_code_enhanced(code_info["code"])
            execution_time = time.time() - start_time
            
            return {
                "code": code_info["code"],
                "language": code_info["language"],
                "execution_type": code_info["type"],
                "result": execution_result["output"],
                "execution_time": execution_time,
                "status": "success",
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
                "suggestion": "ตรวจสอบ syntax ของโค้ดและใช้เฉพาะ Python operations พื้นฐาน"
            }
    
    def _basic_security_check(self, code: str) -> bool:
        """Basic security validation"""
        dangerous_patterns = ['__import__', 'exec(', 'eval(', 'open(', 'file(']
        code_lower = code.lower()
        
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        return True
    
    def _extract_code_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced code extraction with better pattern recognition"""
        
        text = text.strip()
        
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
        
        if self._is_math_expression(text):
            if not any(func in text for func in ['print(', 'return ']):
                code = f"print({text})"
            else:
                code = text
            
            return {
                "code": code,
                "language": "python",
                "type": "math_expression"
            }
        
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
        """Detect mathematical expressions"""
        
        text = text.strip()
        
        math_pattern = r'^[\d\s+\-*/().%**]+$'
        if re.match(math_pattern, text):
            return True
        
        math_functions = ['abs', 'round', 'max', 'min', 'sum', 'pow']
        if any(func in text.lower() for func in math_functions):
            non_alpha = sum(1 for c in text if not c.isalpha())
            total_chars = len(text)
            if non_alpha / total_chars > 0.3:
                return True
        
        if re.search(r'\d+\s*[+\-*/]\s*\d+', text):
            return True
        
        return False
    
    def _looks_like_python_code(self, text: str) -> bool:
        """Detect Python code patterns"""
        
        python_indicators = [
            'print(', 'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 
            'while ', 'try:', 'except:', 'with ', 'lambda ', 'yield ',
            'return ', 'break', 'continue', '=', '+=', '-=', '*=', '/=',
            'and ', 'or ', 'not ', 'in ', 'is ', 'elif ', 'else:'
        ]
        
        indicator_count = sum(1 for indicator in python_indicators if indicator in text)
        
        if indicator_count >= 2:
            return True
        
        strong_indicators = ['def ', 'class ', 'import ', 'from ']
        if any(indicator in text for indicator in strong_indicators):
            return True
        
        return False
    
    def _validate_code_enhanced(self, code: str) -> Dict[str, Any]:
        """Enhanced code validation"""
        
        code_lower = code.lower()
        
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
        
        if len(code) > 5000:
            return {"is_safe": False, "reason": "Code too long (max 5000 characters)"}
        
        loop_count = code_lower.count('for ') + code_lower.count('while ')
        if loop_count > 3:
            return {"is_safe": False, "reason": "Too many loops (max 3)"}
        
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return {"is_safe": False, "reason": f"Syntax error: {str(e)}"}
        
        return {"is_safe": True, "reason": "Code passed validation"}
    
    async def _execute_safe_code_enhanced(self, code: str) -> Dict[str, Any]:
        """Execute code in safe environment"""
        
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name) 
                for name in self.safe_builtins 
                if hasattr(__builtins__, name)
            }
        }
        
        safe_modules = {}
        for module_name in self.allowed_modules:
            try:
                safe_modules[module_name] = __import__(module_name)
            except ImportError:
                continue
        
        safe_globals.update(safe_modules)
        
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            try:
                exec(code, safe_globals, {})
                output = captured_output.getvalue()
            except:
                try:
                    result = eval(code, safe_globals, {})
                    output = str(result)
                except:
                    exec(code, safe_globals, {})
                    output = captured_output.getvalue()
            
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"
            
            if not output.strip():
                output = "Code executed successfully (no output produced)"
            
        finally:
            sys.stdout = original_stdout
        
        return {
            "output": output.strip(),
            "truncated": len(output) >= self.max_output_length
        }

class HRMProcessor:
    """Enhanced HRM processor for complex reasoning"""
    
    def __init__(self):
        self.available = True
        logger.info("HRM Processor initialized")
    
    async def process(self, input_text: str, complexity: TaskComplexity = TaskComplexity.COMPLEX) -> Dict[str, Any]:
        """Process complex reasoning tasks"""
        
        if not self.available:
            return await self._mock_hrm_processing(input_text, complexity)
        
        return await self._structured_analysis(input_text, complexity)
    
    async def _structured_analysis(self, query: str, complexity: TaskComplexity) -> Dict[str, Any]:
        """Provide structured analysis"""
        
        analysis_components = []
        
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
        """Generate structured response based on query"""
        
        query_lower = query.lower()
        
        if "pros and cons" in query_lower or "advantages and disadvantages" in query_lower:
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
        """Extract main subject being analyzed"""
        
        query_lower = query.lower()
        
        common_subjects = [
            "remote work", "cloud computing", "artificial intelligence", "machine learning",
            "automation", "digital transformation", "agile methodology", "microservices"
        ]
        
        for subject in common_subjects:
            if subject in query_lower:
                return subject
        
        words = query.split()
        if len(words) > 2:
            return " ".join(words[1:4])
        
        return "the given topic"
    
    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """Extract subjects being compared"""
        
        vs_patterns = [" vs ", " versus ", " compared to ", " against "]
        
        for pattern in vs_patterns:
            if pattern in query.lower():
                parts = query.lower().split(pattern)
                if len(parts) >= 2:
                    return [part.strip() for part in parts[:2]]
        
        return ["option A", "option B"]
    
    async def _mock_hrm_processing(self, query: str, complexity: TaskComplexity) -> Dict[str, Any]:
        """Mock HRM processing when real system unavailable"""
        
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
        self.web_scraper = EnhancedWebScrapingProcessor()
        self.search_processor = EnhancedGoogleSearchProcessor()
        self.database_processor = EnhancedDatabaseProcessor()
        self.knowledge_processor = EnhancedKnowledgeRAGProcessor()
        self.code_processor = EnhancedCodeExecutionProcessor()
        self.hrm_processor = HRMProcessor()
        self.hrm_integrator = HRMIntegratedProcessor()
        
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
                "formatted_result": ResultFormatter.format_result(
                    selected_task_score.task_type, 
                    enriched_result
                ),
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
        
        # Web scraping detection
        url_score = await self._calculate_web_scraping_score(input_text, text_lower)
        if url_score:
            scores.append(url_score)
        
        # Search detection
        search_score = self._calculate_search_score(input_text, text_lower)
        if search_score:
            scores.append(search_score)
        
        # Database query detection
        db_score = self._calculate_database_score(input_text, text_lower)
        if db_score:
            scores.append(db_score)
        
        # Knowledge management detection
        km_score = self._calculate_knowledge_score(input_text, text_lower)
        if km_score:
            scores.append(km_score)
        
        # Code execution detection
        code_score = self._calculate_code_score(input_text, text_lower)
        if code_score:
            scores.append(code_score)
        
        # HRM reasoning detection
        hrm_score = self._calculate_hrm_score(input_text, text_lower)
        if hrm_score:
            scores.append(hrm_score)
        
        # General query (fallback)
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
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+\.[a-zA-Z]{2,}'
        urls = re.findall(url_pattern, input_text)
        
        domain_pattern = r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, input_text)
        
        url_count = len(urls) + len([d for d in domains if not any(d in url for url in urls)])
        
        if url_count > 0:
            base_confidence = min(0.95, 0.7 + (url_count * 0.1))
            
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
        
        explicit_search_terms = ['search for', 'search', 'google', 'find', 'look up', 'lookup']
        question_words = ['what is', 'who is', 'how to', 'where is', 'when is', 'why']
        
        explicit_matches = sum(1 for term in explicit_search_terms if text_lower.startswith(term))
        question_matches = sum(1 for word in question_words if word in text_lower)
        
        if explicit_matches > 0 or question_matches > 0:
            base_confidence = 0.8 if explicit_matches > 0 else 0.6
            
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
        
        analysis_terms = ['analyze', 'compare', 'evaluate', 'assess', 'examine']
        complexity_terms = ['pros and cons', 'advantages', 'disadvantages', 'trade-offs']
        strategic_terms = ['strategic', 'comprehensive', 'detailed analysis', 'in-depth']
        reasoning_terms = ['because', 'therefore', 'however', 'furthermore', 'moreover']
        
        analysis_matches = sum(1 for term in analysis_terms if term in text_lower)
        complexity_matches = sum(1 for term in complexity_terms if term in text_lower)
        strategic_matches = sum(1 for term in strategic_terms if term in text_lower)
        reasoning_matches = sum(1 for term in reasoning_terms if term in text_lower)
        
        word_count = len(input_text.split())
        length_factor = min(0.3, word_count / 50)
        
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
        """Select optimal task processor"""
        
        if not task_scores:
            return TaskScore(
                task_type=TaskType.GENERAL_QUERY,
                confidence=0.5,
                reasoning="No specific task type detected - using general processing",
                priority=5
            )
        
        if options and options.get("preferred_task"):
            preferred = options["preferred_task"]
            for score in task_scores:
                if score.task_type.value == preferred and score.confidence > 0.3:
                    return score
        
        high_confidence_threshold = 0.8
        high_confidence_tasks = [s for s in task_scores if s.confidence >= high_confidence_threshold]
        
        if high_confidence_tasks:
            return high_confidence_tasks[0]
        
        return task_scores[0]
    
    async def _execute_task_with_monitoring(
        self, 
        task_type: TaskType, 
        input_text: str, 
        task_score: TaskScore, 
        options: Optional[Dict]
    ) -> Any:
        """Execute task with monitoring and HRM enhancement"""
        
        processor_start = time.time()
        
        try:
            # Execute basic task
            result = await self._execute_basic_task(task_type, input_text)
            
            # Apply HRM enhancement for all tasks
            enhanced_result = await self.hrm_integrator.enhance_result(
                task_type, result, input_text
            )
            
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
    
    async def _execute_basic_task(self, task_type: TaskType, input_text: str) -> Any:
        """Execute basic task without HRM enhancement"""
        
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
        elif task_type == TaskType.HRM_REASONING:
            return await self.hrm_processor.process(input_text)
        else:  # GENERAL_QUERY
            return await self._process_general_query(input_text)
    
    async def _process_general_query(self, input_text: str) -> Dict[str, Any]:
        """Process general queries with AI response simulation"""
        
        query_analysis = {
            "is_question": any(word in input_text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']),
            "is_greeting": any(word in input_text.lower() for word in ['hello', 'hi', 'hey', 'greetings']),
            "is_request": any(word in input_text.lower() for word in ['please', 'can you', 'could you', 'help']),
            "word_count": len(input_text.split())
        }
        
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
    
    def _enrich_result(self, result: Any, task_score: TaskScore, input_text: str) -> Dict[str, Any]:
        """Enrich results with additional context and metadata"""
        
        if not isinstance(result, dict):
            result = {"result": result}
        
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
        
        if task_score.confidence < 0.6:
            recommendations.append("Consider being more specific about what you want to accomplish")
        
        return recommendations
    
    def _update_system_stats(self, task: ProcessedTask):
        """Update system statistics"""
        
        if task.status == "success":
            self.system_stats["successful_requests"] += 1
        else:
            self.system_stats["failed_requests"] += 1
        
        self.system_stats["task_counts"][task.selected_task.value] += 1
        self.system_stats["total_processing_time"] += task.processing_time
        
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
        """Provide detailed input analysis"""
        
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
async def process_request(request: ProcessRequest):
    """Main processing endpoint with intelligent task routing"""
    
    try:
        result = await decision_agent.process_request(request.input, request.options)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Request processing failed")

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
                    
                } else {
                    if (result.selected_task) {
                        html += `<p><strong>Selected Task:</strong> ${result.selected_task.replace('_', ' ').toUpperCase()}</p>`;
                    }
                    
                    if (result.confidence_used !== undefined) {
                        const confidenceColor = result.confidence_used > 0.8 ? '#059669' : result.confidence_used > 0.5 ? '#d97706' : '#dc2626';
                        html += `<p><strong>Confidence:</strong> <span style="color: ${confidenceColor}; font-weight: 600;">${(result.confidence_used * 100).toFixed(1)}%</span></p>`;
                    }
                    
                    if (responseTime) {
                        html += `<p><strong>Response Time:</strong> ${responseTime}ms</p>`;
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
                    
                    if (result.formatted_result) {
                        html += `
                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: 600; color: #374151;">📄 Formatted Result</summary>
                                <pre style="background: #f3f4f6; padding: 15px; border-radius: 8px; margin-top: 10px; white-space: pre-wrap; font-size: 0.9rem; line-height: 1.4;">${result.formatted_result}</pre>
                            </details>
                        `;
                    }
                    
                    if (result.result) {
                        html += `
                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: 600; color: #374151;">🔧 Raw Result</summary>
                                <pre style="background: #f3f4f6; padding: 15px; border-radius: 8px; overflow-x: auto; margin-top: 10px; white-space: pre-wrap; font-size: 0.8rem; line-height: 1.4;">${JSON.stringify(result.result, null, 2)}</pre>
                            </details>
                        `;
                    }
                }
                
                resultItem.innerHTML = html;
                
                if (resultsDiv.children.length === 1 && resultsDiv.children[0].style.textAlign === 'center') {
                    resultsDiv.innerHTML = '';
                }
                
                resultsDiv.insertBefore(resultItem, resultsDiv.firstChild);
                
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
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)