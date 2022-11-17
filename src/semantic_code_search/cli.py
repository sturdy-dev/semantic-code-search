import argparse
import os
import sys
from subprocess import run

from sentence_transformers import SentenceTransformer

from semantic_code_search.embed import do_embed
from semantic_code_search.query import do_query


def git_root(path=None):
    path_params = []
    if path:
        path_params = ['-C', path]
    p = run(['git'] + path_params + ['rev-parse',
            '--show-toplevel'], capture_output=True)
    if p.returncode != 0:
        if not path:
            path = os.getcwd()
        print('{} is not a git repo. Run this in a git repository or specify a path using the -p flag'.format(path))
        sys.exit(1)
    return p.stdout.decode('utf-8').strip()


def embed_func(args):
    model = SentenceTransformer(args.model_name_or_path)
    do_embed(args, model)


def query_func(args):
    model = SentenceTransformer(args.model_name_or_path)
    if len(args.query_text) > 0:
        args.query_text = ' '.join(args.query_text)
    else:
        args.query_text = None
    do_query(args, model)


def main():
    parser = argparse.ArgumentParser(
        prog='sem', description='Search your codebase using natural language')
    parser.add_argument('-p', '--path-to-repo', metavar='PATH', default=git_root(), type=git_root, required=False,
                        help='Path to the root of the git repo to search or embed')
    parser.add_argument('-m', '--model-name-or-path', metavar='MODEL', default='krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net',
                        type=str, required=False, help='Name or path of the model to use')
    parser.add_argument('-d', '--embed', action='store_true', default=False,
                        required=False, help='(Re)create the embeddings index for codebase')
    parser.add_argument('-b', '--batch-size', metavar='BS',
                              type=int, default=32, help='Batch size for embeddings generation')

    parser.add_argument('-x', '--file-extension', metavar='EXT', type=str, required=False,
                              help='File extension filter (e.g. "py" will only return results from Python files)')
    parser.add_argument('-n', '--n-results', metavar='N', type=int,
                        required=False, default=5, help='Number of results to return')
    parser.add_argument('-e', '--editor', choices=[
                        'vscode', 'vim'], default='vscode', required=False, help='Editor to open selected result in')
    parser.set_defaults(func=query_func)
    parser.add_argument('query_text', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.embed:
        embed_func(args)
    else:
        query_func(args)


if __name__ == '__main__':
    main()
