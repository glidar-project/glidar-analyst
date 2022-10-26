
This is the README file for the Convection Analysis Project.
A related publication called IsoTrotter can be found at the following link.

	https://arxiv.org/abs/2008.10301


## Data
You will need to download the sounding data separately and place the extracted data folder into the root file of the project. 

https://github.com/glidar-project/glidar-analyst/releases/download/0.1.0/data.zip


## Installation 

Optionally (recommennded):

I suggest installing the project into a virtual environment. The virtual environment is initialized by calling:

	python -m venv env

After initializing the environment it's necessary to activate it by the command

	env\Scritps\activate 

on Windows or

	source env/bin/activate

on Linux.


The glidar_analyst package can be installed using pip by calling

	pip install .

from the project root folder containing setup.py file.

The application can be executed using

	python -m glidar_analyst

If you used the virtual environment to install the application, you can only run it if the virtual environment is active.

Cheers!
Juraj


