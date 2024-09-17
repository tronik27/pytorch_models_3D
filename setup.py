from setuptools import setup

setup(
    name='pytorch_models_3D',
    version='0.1.0',
    description='Deep learning library for 3D data',
    author='Nikita Trotsenko',
    author_email='tronik27@gmail.com',
    url='https://github.com/tronik27/pytorch_models_3D',
    packages=['pytorch_models_3D'],
    install_requires=[
        'torch',
        'torchmetrics',
        'torchsummary',
        'torchvision',
        'matplotlib',
        'more_itertools',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: deep learning 3D data', 'Topic :: Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)