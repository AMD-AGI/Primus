from setuptools import find_packages, setup


def get_version():
    with open("primus/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().replace('"', "").replace("'", "")
    raise RuntimeError("No version found!")


setup(
    name="primus",
    version=get_version(),
    description="Primus: Unified Training Framework for AMD AI",
    author="AMD AIG Team",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "loguru",
        "wandb",
        "pre-commit",
        "nltk",
        "matplotlib",
        "markdown2",
        "weasyprint",
        "tyro",
        "torchao",
        "blobfile",
        "torchdata>=0.8.0",
        "datasets>=3.6.0",
    ],
    package_data={
        "primus": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ]
    },
    entry_points={
        "console_scripts": [
            "primus=primus.cli.main:main",
        ]
    },
    scripts=[
        "bin/primus-cli",
        "bin/primus-cli-slurm.sh",
        "bin/primus-cli-container.sh",
        "bin/primus-cli-entrypoint.sh",
        "bin/primus-env.sh",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
