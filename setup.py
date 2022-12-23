import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='semantic-code-search',
    version='0.4.0',
    author='Kiril Videlov',
    author_email='kiril@codeball.ai',
    description='Search your codebase with natural language.',
    install_requires=[
                'numpy==1.22.4',
                'prompt_toolkit==3.0.32',
                'Pygments==2.12.0',
                'sentence_transformers==2.2.2',
                'setuptools==62.6.0',
                'torch==1.12.1',
                'tree_sitter==0.20.1',
                'tree_sitter_builds==2022.8.27',
                'tree_sitter_languages==1.5.0',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    url='https://github.com/sturdy-dev/semantic-code-search',
    package_dir={
        "semantic_code_search": "src/semantic_code_search"
    },
    entry_points={
        'console_scripts': [
            'sem=semantic_code_search.cli:main',
        ]
    },
    keywords='semantic code search',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development',
        'Topic :: Utilities',
    ]
)
