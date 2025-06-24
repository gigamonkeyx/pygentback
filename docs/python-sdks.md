# 📦 Installed Python SDKs & Libraries

*Last updated: June 5, 2025*

This document provides a comprehensive overview of all Python SDKs and libraries installed in the PyGent Factory environment, organized by category with detailed information about versions, purposes, and key features.

## 🤖 AI & Machine Learning SDKs

### Core AI Frameworks
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `torch` | 2.5.1+cu121 | PyTorch ML framework | GPU acceleration, neural networks, deep learning |
| `torchvision` | 0.20.1+cu121 | Computer vision for PyTorch | Image processing, pre-trained models |
| `transformers` | 4.52.3 | Hugging Face transformers | Pre-trained models, NLP, tokenization |
| `sentence-transformers` | 4.1.0 | Sentence embeddings | Text similarity, semantic search |
| `scikit-learn` | 1.6.1 | Traditional ML library | Classification, regression, clustering |
| `lightgbm` | 4.6.0 | Gradient boosting framework | Fast training, memory efficient |
| `xgboost` | 3.0.2 | Extreme gradient boosting | High performance, feature importance |
| `optuna` | 4.3.0 | Hyperparameter optimization | AutoML, trial management |

### AI Service SDKs
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `anthropic` | 0.52.0 | Anthropic Claude API | Claude models, constitutional AI |
| `openai` | 1.82.0 | OpenAI API client | GPT models, embeddings, fine-tuning |
| `ollama` | 0.4.9 | Local LLM client | Local model hosting, API interface |
| `langchain` | 0.3.25 | LLM application framework | Chains, agents, memory management |
| `langchain-core` | 0.3.62 | LangChain core components | Base classes, interfaces |
| `langchain-text-splitters` | 0.3.8 | Text processing for LangChain | Document chunking, text splitting |
| `langsmith` | 0.3.42 | LangChain monitoring | Tracing, debugging, evaluation |

## 🔗 Model Context Protocol (MCP)

### MCP Framework
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `mcp` | 1.9.1 | **Official MCP Python SDK** | ✅ **DEMONSTRATED** - Client/server implementation |
| `mcp-scholarly` | 0.1.0 | Scholarly MCP server | Academic research integration |

**MCP Integration Status:**
- ✅ **Context7 MCP Server** - Registered and active for live documentation
- ✅ **Cloudflare MCP Servers** - Documentation, Radar, Browser rendering
- ✅ **Local Filesystem MCP** - File operations and code analysis
- ✅ **Python SDK Demo** - Successfully tested with Context7

## 🌐 Web & API SDKs

### Web Frameworks
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `fastapi` | 0.115.9 | Modern web framework | Async, automatic docs, type hints |
| `flask` | 3.0.3 | Lightweight web framework | Simple, flexible, extensible |
| `starlette` | 0.45.3 | ASGI framework | FastAPI foundation, async support |
| `uvicorn` | 0.34.2 | ASGI server | Production-ready, hot reload |
| `flask-cors` | 6.0.0 | CORS support for Flask | Cross-origin resource sharing |

### HTTP Clients
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `requests` | 2.32.3 | HTTP library | Simple, human-friendly API |
| `httpx` | 0.28.1 | Async HTTP client | HTTP/2, async/await support |
| `aiohttp` | 3.12.2 | Async HTTP client/server | WebSocket support, session management |
| `httptools` | 0.6.4 | HTTP parser | Fast HTTP parsing, uvicorn dependency |

### WebSocket & Real-time
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `websockets` | 14.2 | WebSocket client/server | Pure Python, async support |
| `python-socketio` | 5.13.0 | Socket.IO implementation | Real-time bidirectional communication |
| `python-engineio` | 4.12.1 | Engine.IO implementation | Transport abstraction for Socket.IO |

## 💾 Database & Storage SDKs

### SQL Databases
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `sqlalchemy` | 2.0.41 | SQL toolkit and ORM | Database abstraction, migrations |
| `alembic` | 1.16.1 | Database migration tool | Version control for databases |
| `psycopg2-binary` | 2.9.10 | PostgreSQL adapter | Binary wheels, connection pooling |
| `asyncpg` | 0.30.0 | Async PostgreSQL driver | High performance, async/await |
| `aiosqlite` | 0.21.0 | Async SQLite interface | Local database, async operations |

### NoSQL & Specialized
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `redis` | 6.2.0 | Redis client | Caching, pub/sub, data structures |
| `chromadb` | 1.0.11 | Vector database | Embeddings storage, similarity search |
| `supabase` | 2.15.2 | Supabase client | Real-time database, auth, storage |
| `postgrest` | 1.0.2 | PostgREST client | Auto-generated REST API |
| `pgvector` | 0.4.1 | PostgreSQL vector extension | Vector similarity search |

### Search & Indexing
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `faiss-cpu` | 1.11.0 | Facebook AI similarity search | Efficient similarity search |

## 📊 Data Processing & Analytics

### Core Data Libraries
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `pandas` | 2.2.3 | Data manipulation | DataFrames, data analysis, I/O |
| `numpy` | 2.2.6 | Numerical computing | Arrays, mathematical functions |
| `scipy` | 1.15.3 | Scientific computing | Statistics, optimization, signal processing |

### Visualization
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `matplotlib` | 3.10.3 | Plotting library | Static plots, publication quality |
| `seaborn` | 0.13.2 | Statistical visualization | Beautiful statistical plots |
| `plotly` | 6.1.2 | Interactive plotting | Web-based, interactive charts |
| `altair` | 5.5.0 | Declarative visualization | Grammar of graphics |
| `streamlit` | 1.45.1 | Web app framework | Data apps, interactive dashboards |

### Data Formats
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `pyarrow` | 20.0.0 | Columnar data processing | Parquet, Arrow format |
| `orjson` | 3.10.18 | Fast JSON library | High performance JSON parsing |

## 🧪 Testing & Development SDKs

### Testing Frameworks
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `pytest` | 8.3.5 | Testing framework | Fixtures, parametrization, plugins |
| `pytest-asyncio` | 1.0.0 | Async testing | Test async functions |
| `pytest-cov` | 6.1.1 | Coverage reporting | Code coverage analysis |
| `pytest-mock` | 3.14.1 | Mocking for pytest | Test isolation, dependency mocking |
| `pytest-xdist` | 3.7.0 | Distributed testing | Parallel test execution |
| `pytest-timeout` | 2.4.0 | Test timeouts | Prevent hanging tests |
| `pytest-benchmark` | 5.1.0 | Performance testing | Benchmark test functions |
| `coverage` | 7.8.2 | Code coverage | Coverage measurement |

### Browser & UI Testing
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `playwright` | 1.52.0 | Browser automation | Cross-browser, async, reliable |
| `pytest-playwright` | 0.7.0 | Playwright + pytest | Browser testing integration |
| `selenium` | 4.33.0 | Web automation | WebDriver, cross-browser |

### Code Quality
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `black` | 25.1.0 | Code formatter | Opinionated, consistent formatting |
| `isort` | 6.0.1 | Import sorter | Organize imports automatically |
| `flake8` | 7.2.0 | Linting tool | Style guide enforcement |
| `mypy` | 1.15.0 | Static type checker | Type safety, error detection |
| `pre-commit` | 4.2.0 | Git hooks framework | Automated code quality checks |

## 🚀 Deployment & Infrastructure SDKs

### Container & Orchestration
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `kubernetes` | 32.0.1 | Kubernetes API client | Cluster management, deployments |

### Monitoring & Observability
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `opentelemetry-api` | 1.33.1 | OpenTelemetry API | Distributed tracing |
| `opentelemetry-sdk` | 1.33.1 | OpenTelemetry SDK | Telemetry implementation |
| `opentelemetry-instrumentation-fastapi` | 0.54b1 | FastAPI instrumentation | Automatic tracing |
| `memory-profiler` | 0.61.0 | Memory profiling | Memory usage analysis |

## 📚 Document Processing SDKs

### Text & Document Processing
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `beautifulsoup4` | 4.13.4 | HTML/XML parsing | Web scraping, document parsing |
| `lxml` | 5.4.0 | XML/HTML processing | Fast parsing, XPath support |
| `pypdf2` | 3.0.1 | PDF processing | PDF reading, manipulation |
| `python-docx` | 1.1.2 | Word document processing | DOCX file manipulation |
| `markdown` | 3.8 | Markdown processing | Convert markdown to HTML |

### Research & Academic
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `arxiv` | 2.2.0 | arXiv API client | Academic paper access |
| `scholarly` | 1.7.11 | Google Scholar scraping | Academic search, citations |
| `crossref-commons` | 0.0.7 | Crossref API client | DOI resolution, metadata |
| `bibtexparser` | 1.4.3 | BibTeX parsing | Bibliography management |

## 🔧 Utility & System SDKs

### System & Hardware
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `psutil` | 7.0.0 | System monitoring | CPU, memory, disk, network stats |
| `gputil` | 1.4.0 | GPU monitoring | GPU utilization, memory |
| `nvidia-ml-py3` | 7.352.0 | NVIDIA ML API | GPU management, monitoring |

### GPU Computing
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `cupy-cuda11x` | 13.4.1 | GPU computing (CUDA 11) | NumPy-like GPU arrays |
| `cupy-cuda12x` | 13.4.1 | GPU computing (CUDA 12) | NumPy-like GPU arrays |
| `onnxruntime` | 1.22.0 | ONNX runtime | Model inference, optimization |

### Async & Networking
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `anyio` | 4.9.0 | Async compatibility | Backend-agnostic async library |
| `trio` | 0.30.0 | Async library | Alternative to asyncio |
| `asyncio` | Built-in | Async programming | Event loops, coroutines |

## 🛠️ Configuration & CLI SDKs

### Configuration Management
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `pydantic` | 2.11.5 | Data validation | Type checking, serialization |
| `pydantic-settings` | 2.9.1 | Settings management | Environment variable handling |
| `python-dotenv` | 1.1.0 | Environment variables | .env file support |
| `pyyaml` | 6.0.2 | YAML processing | Configuration files |

### CLI & User Interface
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `typer` | 0.16.0 | CLI framework | Type hints, automatic help |
| `click` | 8.2.1 | CLI toolkit | Command line interfaces |
| `rich` | 14.0.0 | Rich text formatting | Beautiful terminal output |
| `colorlog` | 6.9.0 | Colored logging | Enhanced log readability |

## 📈 Analytics & Metrics SDKs

### Performance & Analytics
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `posthog` | 4.2.0 | Product analytics | Event tracking, user analytics |

## 🔐 Security & Authentication

### Authentication & Security
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `passlib` | 1.7.4 | Password hashing | Secure password storage |
| `bcrypt` | 4.3.0 | Password hashing | bcrypt algorithm |
| `pyjwt` | 2.10.1 | JWT tokens | JSON Web Token handling |

## 📦 Package Management & Build Tools

### Development Tools
| Package | Version | Purpose | Key Features |
|---------|---------|---------|--------------|
| `build` | 1.2.2.post1 | Package building | PEP 517 build backend |
| `setuptools` | 65.5.0 | Package development | Package distribution |
| `pip` | 25.1.1 | Package installer | Python package management |

## 🌟 Key Integration Highlights

### ✅ Successfully Demonstrated
- **MCP Python SDK (1.9.1)** - Full integration with Context7 MCP server
- **Context7 Integration** - Live documentation and code examples
- **Cloudflare MCP Servers** - Remote server integration
- **FastAPI + WebSocket** - Real-time API communication
- **Vector Database** - ChromaDB for embeddings
- **GPU Acceleration** - PyTorch with CUDA support

### 🔧 Production Ready
- **Testing Suite** - Comprehensive testing with pytest
- **Code Quality** - Black, isort, flake8, mypy integration
- **Monitoring** - OpenTelemetry observability
- **Database** - PostgreSQL with async support
- **Deployment** - Kubernetes client for orchestration

### 🎯 Specialized Capabilities
- **Academic Research** - arXiv, Scholarly, Crossref integration
- **Document Processing** - PDF, DOCX, Markdown support
- **Browser Automation** - Playwright and Selenium
- **Real-time Communication** - WebSocket and Socket.IO
- **Machine Learning** - Complete ML pipeline support

---

*This documentation represents the complete SDK ecosystem available in the PyGent Factory environment, providing comprehensive capabilities for AI development, web services, data processing, and system integration.*
