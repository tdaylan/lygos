from setuptools import setup, find_packages

setup(
    name = 'Nano',
    packages = find_packages(),
    version = '1.0',
    description = 'Nerede Noktasal PSF photometry', 
    author = 'Tansu Daylan'
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tdaylan/tcat',
    download_url = 'https://github.com/tdaylan/tcat', 
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    #install_requires=['astrophy>=3'],
    include_package_data = True
    )

