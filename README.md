# FL+Privacy+Utility

## How to install the dependencies

I used Poetry as dependency manager. If you don't have poetry installed you can run:

"""
curl -sSL https://install.python-poetry.org | python3 -
"""

Then, we can add all the dependencies:

- git submodule add https://github.com/lucacorbucci/DPL.git
- cd DPL
- poetry build 
- cd ..
- poetry install 