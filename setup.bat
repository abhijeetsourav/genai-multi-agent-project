@echo off
setlocal enabledelayedexpansion

set PROJECT_NAME=genai-multi-agent
set PYTHON_VERSION=3.11

echo 🚀 Setting up Gen AI Multi-Agent Project...

REM Create conda environment
echo 📦 Creating conda environment: %PROJECT_NAME%
call conda create -n %PROJECT_NAME% python=%PYTHON_VERSION% -y

REM Activate environment
echo ✅ Activating environment...
call conda activate %PROJECT_NAME%

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install core dependencies
echo 📚 Installing core dependencies...
pip install -r requirements.txt

REM Install development dependencies
echo 🛠️  Installing development dependencies...
pip install -r requirements-dev.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file...
    copy .env.example .env
    echo ⚠️  Please update .env with your API keys
)

REM Initialize git if not already initialized
if not exist .git (
    echo 🔧 Initializing git repository...
    git init
    git add .
    git commit -m "Initial commit: Project structure setup"
)

echo ✅ Setup complete!
echo 📌 Next steps:
echo    1. conda activate %PROJECT_NAME%
echo    2. Update .env with your API keys
echo    3. Run: jupyter notebook

endlocal