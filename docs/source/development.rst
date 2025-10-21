Development
===========

Testing
-------
Run the unit tests from the project root in a fresh virtual environment:

.. code-block:: bash

   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -e .[test]
   pytest -q

Documentation
-------------
Build the docs locally:

.. code-block:: bash

   cd docs
   make clean html

Then open ``_build/html/index.html`` in your browser.

Style and contributions
-----------------------
- Use NumPy-style docstrings (processed by ``napoleon``).
- Keep public APIs documented and covered by tests.
- Submit PRs with focused changes and include updates to docs/examples.
