import os
from setuptools import setup, find_packages, Command
import subprocess

class BuildSphinx(Command):
    description = "Build Sphinx documentation."
    user_options = [
        ('builder=', 'b', 'Sphinx builder to use (html, latex)')
    ]

    def initialize_options(self):
        self.builder = 'html'
        self.build_dir = None

    def finalize_options(self):
        # Build directory for Sphinx output.
        self.build_dir = os.path.join(os.path.dirname(__file__), 'docs/_build')

    def run(self):
        # First, automatically run sphinx-apidoc to generate .rst files.
        # This documents only the solve_nivp subpackage.
        from sphinx.ext.apidoc import main as sphinx_apidoc_main
        apidoc_args = [
            '--force',         # Overwrite existing .rst files.
            '--module-first',  # Put module documentation before submodule docs.
            '-o', os.path.join('docs', 'source'),  # Output directory for the .rst files.
            'solve_nivp'       # Path to the package to document.
        ]
        sphinx_apidoc_main(apidoc_args)

        # Now, build the documentation using Sphinx.
        from sphinx.cmd.build import main as sphinx_main
        args = [
            '-b', self.builder,
            os.path.join('docs', 'source'),
            os.path.join(self.build_dir, self.builder)
        ]
        errno = sphinx_main(args)
        if errno:
            raise SystemExit(errno)

        # For LaTeX, optionally compile to PDF.
        if self.builder == 'latex':
            latex_dir = os.path.join(self.build_dir, 'latex')
            # This assumes you have a Makefile in your LaTeX output directory.
            errno = subprocess.call(['make', 'all-pdf'], cwd=latex_dir)
            if errno:
                raise SystemExit(errno)
            print("PDF generated in:", os.path.join(latex_dir, 'Documentation.pdf'))

setup(
    name="solve_nivp",
    version="0.1.0",
    packages=find_packages(),  # automatically discovers packages
    description="A solver package for implicit ODEs and projection-based solvers",
    cmdclass={'build_sphinx': BuildSphinx},
)
