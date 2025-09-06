#!/usr/bin/env python3
"""
Setup and Run Script for Decision Agent Fixed
Creates database from schema and starts the application
"""

import os
import sys
import sqlite3
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if required packages are installed"""
    
    logger.info("🔍 Checking requirements...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'openai',
        'requests',
        'beautifulsoup4',
        'python-dotenv',
        'aiohttp'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'beautifulsoup4':
                try:
                    __import__('bs4')
                except ImportError:
                    missing_packages.append(package)
            else:
                missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_packages)
            logger.info("✅ Packages installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install packages: {e}")
            return False
    else:
        logger.info("✅ All required packages are installed")
    
    return True

def setup_environment():
    """Setup environment and check configuration"""
    
    logger.info("🔧 Setting up environment...")
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    logger.info(f"📁 Data directory: {data_dir.absolute()}")
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️ .env file not found. Creating template...")
        
        env_template = """# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=true
ENVIRONMENT=development

# Azure OpenAI Configuration (REQUIRED for real API calls)
AZURE_SUBSCRIPTION_KEY=your-azure-api-key-here
AZURE_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
GPT5_MINI_DEPLOYMENT=gpt-5-mini
GPT5_NANO_DEPLOYMENT=gpt-5-nano

# Database Configuration
DATABASE_PATH=./data/decision_agent.db
ENABLE_DATABASE_LOGGING=false

# Other Configuration
SCRAPING_TIMEOUT=10
CODE_EXECUTION_TIMEOUT=5
ENABLE_CODE_EXECUTION=true
MAX_PROCESSING_HISTORY=1000
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        logger.info(f"📝 Created .env template: {env_file.absolute()}")
        logger.warning("🔑 Please edit .env file with your Azure OpenAI credentials for real API calls")
    else:
        logger.info("✅ .env file found")
    
    return True

def create_database():
    """Create database from schema file"""
    
    logger.info("🗄️ Setting up database...")
    
    db_path = Path("./data/decision_agent.db")
    schema_file = Path("database_schema.sql")
    
    if not schema_file.exists():
        logger.error(f"❌ Database schema file not found: {schema_file}")
        logger.info("Creating minimal database structure...")
        
        # Create minimal database
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    department_id INTEGER,
                    position TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 3,
                    assigned_to INTEGER,
                    estimated_hours DECIMAL(5,2),
                    due_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (assigned_to) REFERENCES users (id)
                );
                
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'active',
                    budget DECIMAL(15,2),
                    manager_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (manager_id) REFERENCES users (id)
                );
                
                -- Insert sample data
                INSERT OR IGNORE INTO users (full_name, email, position) VALUES
                ('Alice Johnson', 'alice@company.com', 'Senior Developer'),
                ('Bob Smith', 'bob@company.com', 'Marketing Manager'),
                ('Carol Davis', 'carol@company.com', 'Sales Representative'),
                ('David Wilson', 'david@company.com', 'Support Specialist'),
                ('Eva Martinez', 'eva@company.com', 'DevOps Engineer');
                
                INSERT OR IGNORE INTO projects (name, description, status, budget, manager_id) VALUES
                ('Customer Portal', 'New customer portal development', 'active', 250000.00, 1),
                ('Mobile App', 'iOS and Android mobile application', 'planning', 400000.00, 1),
                ('Data Analytics', 'Business intelligence platform', 'active', 300000.00, 2);
                
                INSERT OR IGNORE INTO tasks (title, description, status, assigned_to, estimated_hours, due_date) VALUES
                ('API Development', 'Develop REST APIs for portal', 'in_progress', 1, 40.0, '2024-03-30'),
                ('UI Design', 'Design user interface mockups', 'completed', 3, 24.0, '2024-02-28'),
                ('Database Setup', 'Configure production database', 'pending', 5, 16.0, '2024-04-15'),
                ('Testing Framework', 'Set up automated testing', 'in_progress', 1, 32.0, '2024-04-01'),
                ('Documentation', 'Create user documentation', 'pending', 4, 20.0, '2024-04-30');
            """)
            conn.commit()
            logger.info("✅ Minimal database created with sample data")
            
        except Exception as e:
            logger.error(f"❌ Failed to create minimal database: {e}")
            return False
        finally:
            conn.close()
    
    else:
        # Use full schema file
        logger.info(f"📊 Using schema file: {schema_file}")
        
        if db_path.exists():
            logger.info("🔄 Existing database found, backing up...")
            backup_path = db_path.with_suffix('.backup.db')
            if backup_path.exists():
                backup_path.unlink()
            db_path.rename(backup_path)
            logger.info(f"💾 Database backed up to: {backup_path}")
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            conn = sqlite3.connect(db_path)
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
            
            logger.info("✅ Database created successfully from schema file")
            
        except Exception as e:
            logger.error(f"❌ Failed to create database from schema: {e}")
            return False
    
    # Verify database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM tasks")
        task_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM projects")
        project_count = cursor.fetchone()[0]
        
        conn.close()
        
        logger.info(f"📊 Database verification:")
        logger.info(f"   • Users: {user_count}")
        logger.info(f"   • Tasks: {task_count}")
        logger.info(f"   • Projects: {project_count}")
        
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        return False
    
    return True

def check_azure_config():
    """Check Azure OpenAI configuration"""
    
    logger.info("🔑 Checking Azure OpenAI configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
    
    if api_key and endpoint:
        logger.info("✅ Azure OpenAI configuration found")
        logger.info(f"   • Endpoint: {endpoint}")
        logger.info("🚀 Real API calls will be used")
        return True
    else:
        logger.warning("⚠️ Azure OpenAI not configured")
        logger.warning("   • AZURE_SUBSCRIPTION_KEY: " + ("Set" if api_key else "Missing"))
        logger.warning("   • AZURE_OPENAI_ENDPOINT: " + ("Set" if endpoint else "Missing"))
        logger.info("📝 Edit .env file to enable real API calls")
        logger.info("🔄 System will use fallback mode for now")
        return False

def start_application():
    """Start the Decision Agent application"""
    
    logger.info("🚀 Starting Decision Agent Fixed...")
    
    try:
        # Import and run the application
        import uvicorn
        
        # Use the fixed decision agent
        app_module = "decision_agent_complete_fixed:app"
        
        print("\n" + "="*70)
        print("🎯 Decision Agent Fixed - Real Azure OpenAI Integration")
        print("="*70)
        print(f"🌐 Server: http://localhost:8000")
        print(f"🎨 Demo: http://localhost:8000/demo")
        print(f"📚 API Docs: http://localhost:8000/api/docs")
        print(f"📊 Stats: http://localhost:8000/api/stats")
        print("")
        print("🧠 Architecture:")
        print("• HRM Decision Maker: GPT-5-nano")
        print("• General Worker: GPT-5-mini")
        print("• Code Worker: GPT-5")
        print("• Database: Real SQLite with full schema")
        print("")
        print("Press Ctrl+C to stop")
        print("="*70)
        print("")
        
        uvicorn.run(
            app_module,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        return False
    
    return True

def main():
    """Main setup and run function"""
    
    print("🤖 Decision Agent Fixed - Setup & Run")
    print("="*50)
    
    # Step 1: Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Step 2: Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Step 3: Create database
    if not create_database():
        sys.exit(1)
    
    # Step 4: Check Azure configuration
    azure_available = check_azure_config()
    
    print("\n✅ Setup completed successfully!")
    
    if not azure_available:
        print("\n⚠️ IMPORTANT: Azure OpenAI not configured")
        print("   • System will run in fallback mode")
        print("   • Edit .env file to enable real API calls")
        print("   • Restart application after configuration")
    
    print("\n🚀 Starting application...")
    
    # Step 5: Start application
    start_application()

if __name__ == "__main__":
    main()