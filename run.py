import os
import sys
import uvicorn
from config import settings

def main():
    """Main entry point"""
    
    print("🤖 Decision Agent - Complete Task Routing System")
    print("=" * 60)
    print(f"🌐 Server: http://{settings.HOST}:{settings.PORT}")
    print(f"🎨 Demo UI: http://{settings.HOST}:{settings.PORT}/demo")
    print(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/api/docs")
    print(f"📊 Statistics: http://{settings.HOST}:{settings.PORT}/api/stats")
    print("")
    print("🎯 Available Task Types:")
    print("• 📄 Web Scraping (URLs)")
    print("• 🔍 Google Search") 
    print("• 🗄️ Database Queries")
    print("• 📚 Knowledge Management RAG")
    print("• 💻 Code Execution")
    print("• 🧠 HRM Complex Reasoning")
    print("")
    
    # Check configuration
    if not settings.AZURE_SUBSCRIPTION_KEY:
        print("⚠️  Warning: AZURE_SUBSCRIPTION_KEY not set - HRM features will be limited")
        print("   Set your Azure API key in .env file for full functionality")
        print("")
    
    print("🔄 Starting server...")
    print("")
    
    # Start the server
    uvicorn.run(
        "decision_agent_complete:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()
