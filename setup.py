import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glidar-analyst", 
    version="0.1.0",
    author="Juraj Palenik",
    author_email="Juraj.Palenik@uib.no",
    description="Atmospheric convection analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glidar-project/glidar-analyst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'xarray',
        'netCDF4',
        'pandas',
        'metpy',
        'pyproj',
        'pyqt5',
        'pyopengl',
        'pyqtgraph',
        'tqdm'
      ],
    python_requires='>=3.6',

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={  # Optional
        'console_scripts': [
            'glidar-model=glidar_analyst:main',
        ],
    },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        'Site': 'https://glidar-project.github.io/',
    },
)