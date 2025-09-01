# 🤖 Complete Decision Agent

An intelligent task routing system with web scraping, search, database queries, knowledge management, code execution, and hierarchical reasoning capabilities.

## 🎯 Features

- **Web Scraping** - Extract content from URLs automatically
- **Google Search** - Web search with DuckDuckGo integration  
- **Database Queries** - SQL and natural language database access
- **Knowledge RAG** - Company knowledge retrieval system
- **Code Execution** - Safe Python code execution environment
- **HRM Reasoning** - Hierarchical reasoning with Azure OpenAI
- **Confidence Scoring** - Intelligent task selection based on confidence
- **Interactive Demo** - Mobile-responsive web interface

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env` file and update with your settings:

```bash
# Required for HRM features
AZURE_SUBSCRIPTION_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint-url

# Optional configurations
DEBUG=true
PORT=8000
```

### 3. Run the Application

```bash
python decision_agent_complete.py
```

Or with uvicorn:

```bash
uvicorn decision_agent_complete:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Demo

- **Main Interface:** http://localhost:8000
- **Interactive Demo:** http://localhost:8000/demo  
- **API Documentation:** http://localhost:8000/api/docs
- **Statistics:** http://localhost:8000/api/stats

## 📋 Example Usage

### Web Scraping
```
https://example.com
```

### Search
```
search for latest AI trends
```

### Database Query  
```
show all users
```

### Knowledge RAG
```
what is our security policy?
```

### Code Execution
```
calculate: 15 * 37 + 128
```

### Complex Analysis
```
analyze pros and cons of remote work vs office work
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `DEBUG` | Debug mode | false |
| `AZURE_SUBSCRIPTION_KEY` | Azure API key | "" |
| `DATABASE_PATH` | SQLite database path | "./data/decision_agent.db" |
| `SCRAPING_TIMEOUT` | Web scraping timeout | 10 |
| `CODE_EXECUTION_TIMEOUT` | Code execution timeout | 5 |

### Task Processors

Each task type has configurable confidence thresholds and processing parameters. The system automatically selects the best processor based on confidence scores.

## 📊 API Endpoints

### Core Endpoints

- `POST /api/process` - Process requests with task routing
- `POST /api/analyze` - Analyze input confidence scores  
- `GET /api/stats` - Processing statistics
- `GET /api/history` - Task processing history
- `GET /api/health` - Health check

### Request Format

```json
{
  "input": "your request here",
  "options": {
    "timeout": 30,
    "max_results": 10
  }
}
```

### Response Format

```json
{
  "task_id": "uuid",
  "status": "success",
  "selected_task": "web_scraping",
  "confidence_used": 0.95,
  "reasoning": "URLs detected in input",
  "result": {...},
  "processing_time": 1.23
}
```

## 🧠 HRM Integration

When complex reasoning tasks are detected, the system uses Hierarchical Reasoning Model with:

- **Workers (GPT-5-mini)** - Detail processing
- **Heads (GPT-5-nano)** - Coordination  
- **Executive (GPT-5-nano)** - Strategic oversight

Configure Azure endpoints in `.env` file for full HRM functionality.

## 🔒 Security

- Safe code execution environment with restricted imports
- SQL injection protection for database queries
- Rate limiting and timeout controls
- Input sanitization and validation

## 📈 Monitoring

The system tracks:

- Total requests processed
- Success/failure rates  
- Average processing times
- Task type distribution
- Confidence score statistics

Access monitoring at `/api/stats` endpoint.

## 🛠 Development

### Project Structure

```
decision-agent/
├── decision_agent_complete.py  # Main application
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── data/                      # Database and logs
```

### Adding New Task Processors

1. Create processor class in `decision_agent_complete.py`
2. Add task type to `TaskType` enum
3. Implement confidence scoring in `analyze_input()`
4. Add execution logic in `_execute_task()`

### Testing

```bash
# Test API endpoints
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{"input": "https://example.com"}'

# Test analysis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"input": "search for AI news"}'
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues and questions:
- Check the [API documentation](http://localhost:8000/api/docs)
- Review [configuration options](#configuration) 
- Test with the [interactive demo](http://localhost:8000/demo)

## 🔄 Changelog

### v2.0.0
- Complete task routing system
- HRM integration with Azure OpenAI
- Interactive web demo interface
- Comprehensive API documentation
- Database and knowledge management

### v1.0.0
- Initial release with basic processors
- Web scraping and search functionality
- Code execution environment
