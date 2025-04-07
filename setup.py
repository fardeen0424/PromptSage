from setuptools import setup, find_packages

setup(
    name="promptsage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0", 
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "nltk>=3.6.0",
        "wordcloud>=1.8.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    author="PromptSage Team",
    author_email="example@email.com",
    description="A systematic framework for optimizing LLM prompt engineering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/promptsage",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)