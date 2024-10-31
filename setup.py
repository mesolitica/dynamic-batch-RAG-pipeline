import setuptools


__packagename__ = 'dynamic-batch-rag-pipeline'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8',
    description='Dynamic batching for Document Layout and OCR, suitable for RAG',
    author='huseinzol05',
    url='https://github.com/mesolitica/dynamic-batch-rag-pipeline',
)