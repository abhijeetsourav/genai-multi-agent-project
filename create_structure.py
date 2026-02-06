#!/usr/bin/env python3
"""
Automated Project Structure Creator for Gen AI Multi-Agent Project
Best practices: Separation of concerns, modularity, scalability
"""

import os
from pathlib import Path
from typing import Dict, List

def create_file(filepath: Path, content: str = "") -> None:
    """Create a file with optional content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {filepath}")

def create_directory(dirpath: Path) -> None:
    """Create a directory."""
    dirpath.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created: {dirpath}")

def get_project_structure() -> Dict[str, any]:
    """Define the complete project structure."""
    return {
        # Source code
        "src": {
            "__init__.py": "",
            "agents": {
                "__init__.py": "",
                "base_agent.py": AGENT_BASE_TEMPLATE,
                "researcher_agent.py": "",
                "writer_agent.py": "",
                "reviewer_agent.py": "",
            },
            "tools": {
                "__init__.py": "",
                "web_search.py": "",
                "document_processor.py": "",
                "vector_store.py": VECTOR_STORE_TEMPLATE,
            },
            "utils": {
                "__init__.py": "",
                "logger.py": LOGGER_TEMPLATE,
                "config.py": CONFIG_TEMPLATE,
                "helpers.py": "",
            },
            "prompts": {
                "__init__.py": "",
                "templates.py": PROMPT_TEMPLATE,
            },
            "memory": {
                "__init__.py": "",
                "conversation_memory.py": "",
                "vector_memory.py": "",
            },
        },
        
        # Notebooks
        "notebooks": {
            "01_setup_and_test.ipynb": "",
            "02_agent_development.ipynb": "",
            "03_multi_agent_orchestration.ipynb": "",
            "04_evaluation.ipynb": "",
            "README.md": NOTEBOOKS_README,
        },
        
        # Tests
        "tests": {
            "__init__.py": "",
            "test_agents.py": TEST_TEMPLATE,
            "test_tools.py": "",
            "test_utils.py": "",
            "conftest.py": CONFTEST_TEMPLATE,
        },
        
        # Data directories
        "data": {
            "raw": {},
            "processed": {},
            "embeddings": {},
            ".gitkeep": "",
        },
        
        # Configuration
        "config": {
            "agent_configs.yaml": AGENT_CONFIG_YAML,
            "model_configs.yaml": MODEL_CONFIG_YAML,
        },
        
        # Documentation
        "docs": {
            "architecture.md": ARCHITECTURE_DOC,
            "api_reference.md": "",
            "deployment.md": "",
        },
        
        # Scripts
        "scripts": {
            "run_agents.py": RUN_AGENTS_TEMPLATE,
            "evaluate.py": "",
        },
        
        # Logs
        "logs": {
            ".gitkeep": "",
        },
        
        # Output
        "output": {
            ".gitkeep": "",
        },
    }

def create_structure(base_path: Path, structure: Dict) -> None:
    """Recursively create project structure."""
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # It's a directory
            create_directory(path)
            create_structure(path, content)
        else:
            # It's a file
            create_file(path, content)

# File Templates

AGENT_BASE_TEMPLATE = '''"""
Base Agent Class for Multi-Agent System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        name: str,
        llm: Any,
        tools: Optional[list] = None,
        verbose: bool = False
    ):
        self.name = name
        self.llm = llm
        self.tools = tools or []
        self.verbose = verbose
        logger.info(f"Initialized {self.name} agent")
    
    @abstractmethod
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute the agent's main task."""
        pass
    
    def log(self, message: str) -> None:
        """Log agent activity."""
        if self.verbose:
            logger.info(f"[{self.name}] {message}")
'''

VECTOR_STORE_TEMPLATE = '''"""
Vector Store Management for Document Embeddings
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage document embeddings and retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings",
        embedding_model: str = "text-embedding-ada-002"
    ):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        
    def create_vectorstore(
        self,
        documents: List[Dict[str, Any]]
    ) -> Chroma:
        """Create vector store from documents."""
        try:
            langchain_docs = self._prepare_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Created vector store with {len(langchain_docs)} documents")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def _prepare_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """Convert raw documents to LangChain Documents."""
        langchain_docs = []
        
        for idx, doc in enumerate(documents):
            if not doc.get("text") or not doc.get("id"):
                logger.warning(f"Skipping invalid document at index {idx}")
                continue
                
            langchain_docs.append(
                Document(
                    page_content=doc["text"],
                    metadata={
                        "doc_id": doc["id"],
                        "source": doc.get("source", "unknown"),
                        "index": idx
                    }
                )
            )
        
        return langchain_docs
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """Search for similar documents."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)
'''

LOGGER_TEMPLATE = '''"""
Logging Configuration
"""
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "genai_project",
    level: int = logging.INFO,
    log_file: str = "logs/app.log"
) -> logging.Logger:
    """Setup project logger with file and console handlers."""
    
    # Create logs directory
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
'''

CONFIG_TEMPLATE = '''"""
Configuration Management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # API Keys
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Grok Configuration
    GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta")
    
    # Model settings
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    
    # Vector store
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    VECTOR_STORE_PATH = DATA_DIR / "embeddings"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.GROK_API_KEY:
            raise ValueError("GROK_API_KEY not found in environment")
'''

PROMPT_TEMPLATE = '''"""
Prompt Templates for Agents
"""

RESEARCHER_PROMPT = """
You are a research assistant agent. Your role is to:
1. Analyze the given query
2. Search for relevant information
3. Synthesize findings into a coherent summary

Query: {query}
Context: {context}

Provide a detailed research summary.
"""

WRITER_PROMPT = """
You are a content writer agent. Your role is to:
1. Take research findings
2. Create well-structured, engaging content
3. Maintain consistent tone and style

Research Input: {research}
Style Guide: {style}

Create high-quality content based on the research.
"""

REVIEWER_PROMPT = """
You are a quality reviewer agent. Your role is to:
1. Review content for accuracy
2. Check grammar and style
3. Suggest improvements

Content to Review: {content}

Provide detailed feedback and suggestions.
"""
'''

TEST_TEMPLATE = '''"""
Unit Tests for Agents
"""
import pytest
from src.agents.base_agent import BaseAgent


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        # Add your tests here
        pass
    
    def test_agent_run(self):
        """Test agent run method."""
        # Add your tests here
        pass
'''

CONFTEST_TEMPLATE = '''"""
Pytest Configuration and Fixtures
"""
import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return {
        "api_key": os.getenv("GROK_API_KEY"),
        "model": "grok-beta"
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "1", "text": "Sample document 1"},
        {"id": "2", "text": "Sample document 2"},
    ]
'''

AGENT_CONFIG_YAML = '''# Agent Configurations

researcher:
  name: "Research Agent"
  model: "grok-beta"
  temperature: 0.3
  max_tokens: 2000
  tools:
    - web_search
    - document_retrieval

writer:
  name: "Writer Agent"
  model: "grok-beta"
  temperature: 0.7
  max_tokens: 3000
  tools:
    - grammar_check

reviewer:
  name: "Reviewer Agent"
  model: "grok-beta"
  temperature: 0.2
  max_tokens: 2000
'''

MODEL_CONFIG_YAML = '''# Model Configurations

grok:
  base_url: "https://api.x.ai/v1"
  model: "grok-beta"
  default_temperature: 0.7
  default_max_tokens: 2000
  timeout: 60

embeddings:
  model: "text-embedding-ada-002"
  dimensions: 1536
'''

ARCHITECTURE_DOC = '''# Project Architecture

## Overview
Multi-agent system for [your use case]

## Components

### Agents
- **Research Agent**: Gathers and analyzes information
- **Writer Agent**: Creates content based on research
- **Reviewer Agent**: Quality assurance and feedback

### Tools
- Web search integration
- Document processing
- Vector store for semantic search

### Memory
- Conversation history
- Vector-based long-term memory

## Data Flow
1. User input → Research Agent
2. Research Agent → Vector Store
3. Findings → Writer Agent
4. Content → Reviewer Agent
5. Final output → User
'''

NOTEBOOKS_README = '''# Notebooks Guide

## Organization

1. **01_setup_and_test.ipynb**: Environment setup and API testing
2. **02_agent_development.ipynb**: Individual agent development and testing
3. **03_multi_agent_orchestration.ipynb**: Multi-agent workflow
4. **04_evaluation.ipynb**: Performance evaluation and metrics

## Best Practices
- Run notebooks in order
- Clear outputs before committing
- Document experiments
'''

RUN_AGENTS_TEMPLATE = '''#!/usr/bin/env python3
"""
Main script to run multi-agent system
"""
import argparse
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger()


def main(task: str):
    """Run multi-agent system."""
    logger.info(f"Starting multi-agent system with task: {task}")
    
    # Initialize agents
    # Run orchestration
    # Return results
    
    logger.info("Task completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent system")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    args = parser.parse_args()
    
    main(args.task)
'''

def main():
    """Main function to create project structure."""
    print("🏗️  Creating Gen AI Multi-Agent Project Structure...\n")
    
    base_path = Path.cwd()
    structure = get_project_structure()
    
    create_structure(base_path, structure)
    
    # Create root-level files
    root_files = {
        ".env.example": ENV_EXAMPLE,
        ".gitignore": GITIGNORE,
        "requirements.txt": REQUIREMENTS,
        "requirements-dev.txt": REQUIREMENTS_DEV,
        "README.md": README,
        "pyproject.toml": PYPROJECT_TOML,
    }
    
    for filename, content in root_files.items():
        create_file(base_path / filename, content)
    
    print("\n✅ Project structure created successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: chmod +x setup.sh (Linux/Mac)")
    print("   2. Run: ./setup.sh (Linux/Mac) or setup.bat (Windows)")
    print("   3. Update .env with your API keys")
    print("   4. Start developing in notebooks/")


# Root-level file contents

ENV_EXAMPLE = '''# Grok Cloud API
GROK_API_KEY=your_grok_api_key_here
GROK_BASE_URL=https://api.x.ai/v1
GROK_MODEL=grok-beta

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
TEMPERATURE=0.7
MAX_TOKENS=2000

# Embeddings
EMBEDDING_MODEL=text-embedding-ada-002

# Environment
ENVIRONMENT=development
'''

GITIGNORE = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Environment
.env
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Data
data/raw/*
data/processed/*
data/embeddings/*
!data/*/.gitkeep

# Logs
logs/*.log
*.log

# Output
output/*
!output/.gitkeep

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
'''

REQUIREMENTS = '''# Core dependencies
langchain==0.1.0
langchain-openai==0.0.5
langchain-community==0.0.20
langchain-chroma==0.1.0

# Vector stores
chromadb==0.4.22

# Utilities
python-dotenv==1.0.0
pydantic==2.5.3
pyyaml==6.0.1
requests==2.31.0

# Data processing
numpy==1.24.3
pandas==2.0.3

# Jupyter
jupyter==1.0.0
ipykernel==6.27.1
notebook==7.0.6

# HTTP client
httpx==0.26.0
'''

REQUIREMENTS_DEV = '''# Development dependencies
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1

# Code quality
black==23.12.1
flake8==7.0.0
isort==5.13.2
mypy==1.8.0

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==2.0.0

# Pre-commit hooks
pre-commit==3.6.0
'''

README = '''# Gen AI Multi-Agent Project

A production-ready multi-agent system using Grok Cloud API and LangChain.

## 🚀 Quick Start

### Setup
\`\`\`bash
# Linux/Mac
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
\`\`\`

### Activate Environment
\`\`\`bash
conda activate genai-multi-agent
\`\`\`

### Configure
1. Copy `.env.example` to `.env`
2. Add your Grok API key and other credentials
3. Update configuration in `config/`

### Run
\`\`\`bash
# Launch Jupyter
jupyter notebook

# Or run scripts
python scripts/run_agents.py --task "your task here"
\`\`\`

## 📁 Project Structure
```
├── src/                # Source code
│   ├── agents/        # Agent implementations
│   ├── tools/         # Agent tools
│   ├── utils/         # Utilities
│   └── prompts/       # Prompt templates
├── notebooks/         # Jupyter notebooks
├── tests/            # Unit tests
├── config/           # Configuration files
├── data/             # Data storage
└── docs/             # Documentation
```

## 🧪 Testing

\`\`\`bash
pytest tests/
\`\`\`

## 📚 Documentation

See `docs/` folder for detailed documentation.

## 🤝 Contributing

1. Create feature branch
2. Make changes
3. Run tests
4. Submit PR

## 📝 License

[Your License]
'''

PYPROJECT_TOML = '''[tool.black]
line-length = 88
target-version = ['py311']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
'''

if __name__ == "__main__":
    main()