# Gen AI Multi-Agent Project

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
