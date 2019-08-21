from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ecc",
    version="0.0.1",
    author="Tomoyuki Mano",
    author_email="tmano@m.u-tokyo.ac.jp",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DSPsleeporg/ecc",
    include_package_data=True, # include some resource files, also see MANIFEST.in
    classifiers=[
	    "Programming Language :: Python :: 3",
	    "License :: OSI Approved :: MIT License",
	    "Operating System :: OS Independent",
	]
)