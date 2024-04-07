"""Python setup.py for PROJECTNAME package"""
import io
import os
from setuptools import find_packages, setup

PACKAGE_NAME = "fuse"
SOURCE_DIRECTORY = 'src'

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content

requirements = [
    "pathlib2",
    "wandb",
    "hydra-core"
]

setup(
    name=PACKAGE_NAME,
    # version=read("project_name", "VERSION"),
    description="project_description",
    # url="https://github.com/author_name/project_urlname/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=[SOURCE_DIRECTORY], exclude=["tests", ".github"]),
    package_dir={"": SOURCE_DIRECTORY},
    install_requires=requirements,
    # entry_points={
    #     "console_scripts": ["project_name = project_name.__main__:main"]
    # },
    # extras_require={"test": read_requirements("requirements-test.txt")},
)