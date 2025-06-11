import os

# Define the folder structure
folders = [
    ".vscode",
    ".github/workflows",
    "src",
    "notebooks",
    "scripts",
    "tests",
    "data/raw",
    "data/processed",
    "models",
    "reports/figures",
    "config",
    "utils",
    "app",
]

# Define files with their content
files = {
    # IDE Configuration
    os.path.join(
        ".vscode", "settings.json"
    ): """{
    "python.pythonPath": "venv/bin/python",
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "jupyter.notebookFileRoot": "${workspaceFolder}/notebooks"
}""",
    # CI/CD Pipeline (runs on any branch)
    os.path.join(
        ".github", "workflows", "ci-cd.yml"
    ): """name: CI/CD Pipeline
on:
  push:
    branches: # Runs on all branches
  pull_request:
    branches: # Runs on all branches

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
""",
    # Git ignore rules
    ".gitignore": """# Python artifacts
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.env
.venv
venv/
env

# Data files
*.csv
*.parquet
*.pkl
*.db
*.xlsx
data/

# Notebooks
*.ipynb_checkpoints/

# OS files
.DS_Store

# Logs
*.log
""",
    # Requirements with full data science stack
    "requirements.txt": """# Core data analysis
numpy==1.26.4
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
scipy==1.12.0
statsmodels==0.14.1
scikit-learn==1.4.1.post1

# Financial data
yfinance==0.2.37
pandas-datareader==0.10.0

# Jupyter notebook support
jupyter==1.0.0
ipykernel==6.29.0
ipywidgets==8.1.2
nbformat==5.10.2

# Data processing
openpyxl==3.1.2
xlrd==2.0.1

# Dev tools
pytest==8.0.0
black==24.3.0
python-dotenv==1.0.1
""",
    # Main README
    "README.md": """# Data Analysis Project

## Project Structure
- `src/` - Source code package
- `scripts/` - Standalone scripts for data processing
- `notebooks/` - Jupyter notebooks for analysis
- `app/` - Main application package
- `tests/` - Test cases
- `data/` - Raw and processed data
- `models/` - Trained models
- `config/` - Configuration files
- `reports/` - Generated figures and reports

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run tests using `pytest tests/`.
""",
    # Package initializers
    os.path.join("src", "__init__.py"): "# Source code package",
    os.path.join("scripts", "__init__.py"): "# Scripts package",
    os.path.join("notebooks", "__init__.py"): "# Notebooks package",
    os.path.join("app", "__init__.py"): "# Application package",
    # Additional READMEs
    os.path.join(
        "notebooks", "README.md"
    ): "# Analysis Notebooks\n\nPlace Jupyter notebooks here for exploratory analysis",
    os.path.join(
        "scripts", "README.md"
    ): "# Utility Scripts\n\nStandalone scripts for data processing and automation",
    # Sample test file
    os.path.join("tests", "__init__.py"): "",
    os.path.join(
        "tests", "test_sample.py"
    ): """def test_example():
    assert 1 + 1 == 2
""",
    # Configuration
    os.path.join(
        "config", "config.yaml"
    ): """# Project parameters
data_path: data/raw/
model_path: models/
""",
    # Utility code
    os.path.join("utils", "__init__.py"): "",
    os.path.join(
        "utils", "helpers.py"
    ): '''def load_data(path):
    """Load dataset from file"""
    import pandas as pd
    return pd.read_csv(path)
''',
    # Main application file
    "main.py": '''# Main application entry point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Main application function"""
    print("Data Analysis Project")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    
    # Sample data visualization
    data = pd.Series(np.random.randn(1000))
    data.plot(kind='hist', title='Sample Distribution')
    plt.savefig('reports/figures/sample_distribution.png')
    print("Generated sample visualization")

if __name__ == "__main__":
    main()
''',
}


def create_folders_and_files():
    # Create folder structure
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"Created: {folder}/")
        except Exception as e:
            print(f"Error creating folder {folder}: {e}")

    # Create files with content
    for path, content in files.items():
        try:
            # Ensure parent directories exist
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            with open(path, "w") as f:
                f.write(content)
            print(f"Created: {path}")
        except Exception as e:
            print(f"Error creating file {path}: {e}")

    print("\nProject structure created successfully! ✨")
    print("Key features:")
    print("- CI/CD pipeline that runs on ANY branch")
    print("- Full data science stack in requirements.txt")
    print("- Jupyter notebook support included")
    print("- Main.py entry point with sample visualization")
    print("- Package init files in all code directories")


if __name__ == "__main__":
    create_folders_and_files()
