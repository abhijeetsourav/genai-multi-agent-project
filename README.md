# GenAI Multi-Agent Project

A production-grade multi-agent system using GrokCloud API and LangGraph.

## Project Structure
```
genai-multi-agent-project/
├── data/
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed data
│   └── outputs/          # Agent outputs
├── notebooks/            # Jupyter notebooks for experiments
├── src/
│   ├── agents/          # Agent implementations
│   ├── utils/           # Utility functions
│   └── config/          # Configuration files
├── logs/                # Application logs
├── models/              # Saved models/prompts
├── environment.yml      # Conda environment
├── requirements.txt     # Pip requirements
└── .env                 # Environment variables (create from .env.example)
```

## Setup

1. Create conda environment:
```bash
   conda env create -f environment.yml
   conda activate genai-multi-agent
```

2. Configure environment variables:
```bash
   cp .env.example .env
   # Edit .env with your GrokCloud API key
```

3. Start Jupyter:
```bash
   jupyter notebook
```

## Usage

See notebooks/ for examples.
