"""
Setup configuration for local-model-manager package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="local-model-manager",
    version="1.0.0",
    author="Local Model Manager Team",
    author_email="contact@localmodelmanager.dev",
    description="Local model management system optimized for edge AI deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/local-model-manager",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "llama-cpp-python>=0.2.0",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "psutil>=5.9.0",
        "GPUtil>=1.4.0",
        "aiofiles>=23.2.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "local-model-server=local_model_manager.api.server:main",
            "local-model-download=local_model_manager.core.model_manager:main",
        ],
    },
    include_package_data=True,
    package_data={
        "local_model_manager": ["configs/*.yaml"],
    },
    zip_safe=False,
)
