from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repo2vec",
    version="0.1.0",
    author="Md Omar Faruk Rokon",
    author_email="mroko001@ucr.edu",
    description="A tool for creating vector representations of software repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ofrokon/repo2vec",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "repo2vec=cli:main",
        ],
    },
    install_requires=[
        "networkx>=2.5",
        "torch>=2.0.0",
        "torch-geometric>=2.0.0",
        "transformers>=4.30.0",
        "tree-sitter>=0.22.0",
        "typer>=0.9.0",
    ],
    extras_require={
        "languages": [
            "tree-sitter-python",
            "tree-sitter-javascript",
            "tree-sitter-typescript",
            "tree-sitter-java",
            "tree-sitter-go",
            "tree-sitter-rust",
            "tree-sitter-c",
            "tree-sitter-cpp",
            "tree-sitter-c-sharp",
            "tree-sitter-ruby",
        ],
    },
)
