#!/usr/bin/env python3
"""Create anonymous artifact for submission."""

import shutil
import zipfile
from pathlib import Path
import json
import re


def anonymize_file(content: str) -> str:
    """Remove identifying information from file content."""
    # Remove email addresses
    content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     'anonymous@example.com', content)
    
    # Remove common identifying patterns
    content = re.sub(r'(author|Author|AUTHOR).*', 'Author: Anonymous', content)
    content = re.sub(r'(copyright|Copyright|COPYRIGHT).*', 'Copyright: Anonymous', content)
    
    return content


def create_artifact(output_path: str = "mango_artifact.zip"):
    """Create anonymous artifact zip."""
    print("Creating anonymous artifact...")
    
    # Directories to include
    include_dirs = [
        "faircare",
        "tests",
        "scripts",
        "paper"
    ]
    
    # Files to include
    include_files = [
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    # Create temporary directory
    temp_dir = Path("temp_artifact")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Copy directories
    for dir_name in include_dirs:
        src = Path(dir_name)
        if src.exists():
            dst = temp_dir / dir_name
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '.DS_Store', '*.egg-info'
            ))
    
    # Copy and anonymize files
    for file_name in include_files:
        src = Path(file_name)
        if src.exists():
            dst = temp_dir / file_name
            
            # Read and anonymize text files
            if src.suffix in ['.md', '.txt', '.toml', '.yaml', '.yml']:
                with open(src, 'r') as f:
                    content = f.read()
                content = anonymize_file(content)
                with open(dst, 'w') as f:
                    f.write(content)
            else:
                shutil.copy2(src, dst)
    
    # Create README for artifact
    artifact_readme = temp_dir / "ARTIFACT_README.md"
    with open(artifact_readme, 'w') as f:
        f.write("""# MANGO: Fair Federated Learning Framework - Artifact

This is an anonymous artifact for paper submission.

## Installation

```bash
pip install -e .
```

## Quick Test

```bash
pytest tests/
```

## Reproduce Results

```bash
bash scripts/reproduce_tables.sh
```

## Contents

- `faircare/`: Main framework implementation
- `tests/`: Unit tests
- `scripts/`: Reproduction scripts
- `paper/`: Paper generation utilities
""")
    
    # Create zip archive
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    print(f"Artifact created: {output_path}")
    print(f"Size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    create_artifact()
