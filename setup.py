from setuptools import setup, find_packages

setup(
    name='deepfdr',
    version='0.1',
    packages=find_packages(),
    description='FDR Control method using deep unsupervised segmentation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='The repository is maintained by Taehyo Kim',
    author_email='tk2737.nyu.edu',
    url='https://github.com/kimtae55/DeepFDR',
    install_requires=[
        'argparse',
        'numpy',
        'torchinfo',
        'multiprocessing',
        'plotly',
        'dash',
        'dash_slicer',
        'scikit-image',
        'imageio',
        'flask',
        'statsmodels',
        'scipy'
    ],
)