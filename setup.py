from setuptools import setup, find_packages


def readme():
    """
    This function just return the content of README.md
    """
    with open('README.md') as f:
        return f.read()


setup(
    name='pydata',
    version='0.0',
    description='Simple file handler written in Python',
    long_description=readme(),
    classifiers=[
        'Development Status :: 1 - Planning'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords='filehandler iges stl vtk',
    url='https://github.com/mathLab/pydata',
    author='Marco Tezzele, Nicola Demo',
    author_email='marcotez@gmail.com, demo.nicola@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False)
