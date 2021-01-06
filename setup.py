from setuptools import setup, find_packages

setup(
    name = 'lygos',
    packages = find_packages(),
    version = '1.1',
    description = 'PSF modeling of images in time domain astronomy', 
    author = 'Tansu Daylan',
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tdaylan/lygos',
    download_url = 'https://github.com/tdaylan/lygos', 
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    #install_requires=['astrophy>=3'],
    include_package_data = True
    )

