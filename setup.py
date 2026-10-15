from setuptools import find_packages, setup


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
        return lines


setup(
    name="agentcore",
    version="0.1",
    description="Light Library for Building LLM Agent System",
    packages=find_packages(exclude=["test*", "doc*", "example", "benchmark"]),
    python_requires=">=3.10",
    install_requires=get_requires(),
    author="Yaoqi Chen",
    author_email="yaoqi_chen@mail.ustc.edu.cn",
)
