import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="obfuscation-detection",
    version="0.7.2",
    author="Wilson Tang",
    author_email="wilson.tang06@gmail.com",
    description="Python module for obfuscation classification in command line executions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adobe/SI-Obfuscation-Detection",
    project_urls={
        "Bug Tracker": "https://github.com/adobe/SI-Obfuscation-Detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires = ['torch>=1.9.0'],
    python_requires=">=3.6",
    include_package_data=True,
)