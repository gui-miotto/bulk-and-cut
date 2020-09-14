import setuptools, os

HERE = os.path.abspath(os.path.dirname(__file__))
setup_reqs = []
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BulkAndCut",
    version="0.1.0",
    author="Gui Miotto",
    author_email="guilherme.miotto@gmail.com",
    description="Multi-objective optimization for neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl-classroom/final-project-gui-miotto",
    packages=setuptools.find_packages(exclude=['test', 'examples']),
    setup_requires=setup_reqs,
    install_requires=install_reqs,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
