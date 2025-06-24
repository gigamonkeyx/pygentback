#!/usr/bin/env python3
"""
PyGent A2A Client SDK Setup

Official Python client for PyGent Factory A2A Multi-Agent System.
"""

from setuptools import setup, find_packages
import os

# Read version from package
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'pygent_a2a_client', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="pygent-a2a-client",
    version=get_version(),
    author="PyGent Factory Team",
    author_email="support@timpayne.net",
    description="Official Python client for PyGent Factory A2A Multi-Agent System",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/gigamonkeyx/pygentback",
    project_urls={
        "Documentation": "https://docs.timpayne.net/a2a",
        "Source": "https://github.com/gigamonkeyx/pygentback",
        "Tracker": "https://github.com/gigamonkeyx/pygentback/issues",
        "API Reference": "https://api.timpayne.net/a2a/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "asyncio-throttle>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "pandas>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "a2a-client=pygent_a2a_client.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pygent_a2a_client": [
            "py.typed",
            "*.pyi",
        ],
    },
    keywords=[
        "a2a",
        "agent",
        "multi-agent",
        "ai",
        "artificial-intelligence",
        "pygent",
        "client",
        "sdk",
        "api",
        "async",
        "asyncio",
    ],
    zip_safe=False,
)
