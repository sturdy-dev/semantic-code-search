# Semantic Code Search

Search your code from the terminal using natural language.

`sem` is a command line application which allows you to search your git repository using natural language. **Example queries**:

- 'Where are API requests authenticated?'
- 'Saving user objects to the database'
- 'Handling of webhook events'

You will get a prompt with the code snippets that match your query semantically. Pressing `Return` will open the relevant file at the matching location in your editor of choice.

How does this work? In a nutshell, it uses a neural network to generate embeddings of your code and queries. More info [below](#how-it-works).

> NB: All processing is done on your hardware and no data is transmitted to the Internet.

## Installation

You can install `semantic-code-search` via `pip`.

### Pip (MacOS, Linux, Windows)

```bash
pip3 install semantic-code-search
```

## Usage

**TL;DR**

```bash
cd /my/repo
sem embed
sem query 'my query'
```

The command line script is `sem`. You can run `sem --help` to see all available options.

`sem` has two subcommands:

- [`embed`](#running-embed)
- [`query`](#running-query)

Before you can query anything, you have to create 'embeddings' for your repository with the `embed` command. After that, you search with the `query` command.

You need to run `sem` inside a git repository or provide a path to a repo with the `-p` argument.

> NB: The first time you run `sem` it will download an ML model so that it can perform all operations locally.

### Running `embed`

The `embed` subcommand will create information dense representations of functions and methods in your codebase (you can think of these as an 'index').

Depending on the size of your repository, this can take from a couple of seconds to minutes.

Embeddings will be stored in an `.embeddings` file at the root of your repository.

Example:

```bash session
foo@bar:~$ cd /my/repo
foo@bar:~$ sem embed

Embedding 15 functions in 1 batches. This is done once and cached in .embeddings
Batches: 100%|██████████████████████████████████████████| 1/1 [00:05<00:00,  5.05s/it]
```

### Running `query`

This is the subcommand you use for querying. It uses the `.embeddings` file in the root of the repository, so it requires that the `embed` command is ran first.

Example:

```bash session
foo@bar:~$ sem query 'command line parsing'
Go to result for query "command line parsing"
┌────────────────────────────────────────────────────────────────┐
│👉   0.832 src/semantic_code_search/cli.py:32                   │
│                                                                │
│def main():                                                     │
│    parser = argparse.ArgumentParser(                           │
│        prog='sem', description='Search your codebase using natu│
│    parser.add_argument('-p', '--path-to-repo', metavar='PATH', │
│                        help='Path to the root of the git repo t
                                ...
```

Result candidates can be navigated using the `↑` `↓` arrow keys or `vim` bindings. Pressing `return` will open the relevant file at the line of the code snippet in your editor.

#### Opening search results in different editors

By default, `sem` will try to open files using `VSCode`. This is controlled by the `--editor` argument of the `query` subcommand.

- `--editor vscode` opens results in VSCode
- `--editor vim` opens results in Vim

### Command line flags

``` bash
usage: sem [-h] [-p PATH] [-m MODEL] {embed,query} ...

Search your codebase using natural language

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path-to-repo PATH
                        Path to the root of the git repo to search or embed
  -m MODEL, --model-name-or-path MODEL
                        Name or path of the model to use

subcommands:
  {embed,query}
    embed               (Re)create the embeddings index for codebase
    query               Search the codebase using natural language

```

``` bash
usage: sem embed [-h] [-b BS]

optional arguments:
  -h, --help            show this help message and exit
  -b BS, --batch-size BS
                        Batch size for embeddings generation

```

``` bash
usage: sem query [-h] [-x EXT] [-n N] [-e {vscode,vim}] ...

positional arguments:
  query_text

optional arguments:
  -h, --help            show this help message and exit
  -x EXT, --file-extension EXT
                        File extension filter (e.g. "py" will only retrun
                        results from Python files)
  -n N, --n-results N   Number of results to return
  -e {vscode,vim}, --editor {vscode,vim}
                        Editor to open selected result in
```

## How it works

In a nutshell, this application uses a [transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) machine learning model to generate embeddings of methods and functions in your codebase. Embeddings are information dense numerical representations of the semantics of the text/code they represent.

Here is  great blog post by Jay Alammar which explains the concept really nicely:
> <https://jalammar.github.io/illustrated-word2vec/>

When the `embed` subcommand is ran, function and method definitions are first extracted from the source files and then used for sentence embedding. To avoid doing this for every query, the results are compressed and saved in an `.embeddings` file.

When the `query` subcommand is ran, embeddings are generated from the query text. This is then used in a 'nearest neighbor' search to discover function or methods with similar embeddings. We are basically comparing the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between vectors.

### Model

The application uses [sentence transformer](https://www.sbert.net/) model architecture to produce 'sentence' embeddings for functions and queries. The particular model is [krlvi/sentence-t5-base-nlpl-code-x-glue](https://huggingface.co/krlvi/sentence-t5-base-nlpl-code-x-glue) which is based of a [SentenceT5-Base](https://github.com/google-research/t5x_retrieval#released-model-checkpoints) checkpoint with 110M parameters and a pooling layer.

It has been further trained on the [code_x_glue_tc_text_to_code](https://huggingface.co/datasets/code_x_glue_tc_text_to_code) dataset of 'natural language' — 'programming language' pairs.

You can experiment with your own sentence transformer models with the `--model` parameter.

## Bugs and limitations

- Currently the `.embeddings` index is not updated when repository files change. As a temporary workaround, `sem embed` can be re-ran occasionally.
- Supported languages: `{ 'python', 'javascript', 'typescript', 'ruby', 'go', 'rust', 'java' }`
- Supported text editors for opening results in: `{ 'vscode', 'vim' }`

## License

Semantic Code Search is distributed under [AGPL-3.0-only](LICENSE.txt). For Apache-2.0 exceptions — <kiril@codeball.ai>
