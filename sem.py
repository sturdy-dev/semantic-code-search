import torch
from InquirerPy.separator import Separator
from InquirerPy.base.control import Choice
from InquirerPy import inquirer
import gzip
from subprocess import run
import openai
import argparse
import pickle

from tree_sitter import Language, Parser, Tree
import numpy as np
from textwrap import dedent
import os


def get_language_support(language_lib_path='build/my-languages.so'):
    RUBY = Language(language_lib_path, 'ruby')
    GO = Language(language_lib_path, 'go')
    RUST = Language(language_lib_path, 'rust')
    JAVA = Language(language_lib_path, 'java')
    VUE = Language(language_lib_path, 'vue')
    SVELTE = Language(language_lib_path, 'svelte')
    JAVASCRIPT = Language(language_lib_path, 'javascript')
    TYPESCRIPT = Language(language_lib_path, 'typescript')
    PYTHON = Language(language_lib_path, 'python')

    return {
        '.rb': RUBY,
        '.go': GO,
        '.rs': RUST,
        '.java': JAVA,
        '.vue': VUE,
        '.svelte': SVELTE,
        '.js': JAVASCRIPT,
        '.ts': TYPESCRIPT,
        '.py': PYTHON
    }


def traverse_tree(tree: Tree):
    cursor = tree.walk()
    reached_root = False
    while reached_root == False:
        yield cursor.node
        if cursor.goto_first_child():
            continue
        if cursor.goto_next_sibling():
            continue
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True
            if cursor.goto_next_sibling():
                retracing = False


def extract_functions(nodes, fp, file_content, relevant_node_types):
    out = []
    for n in nodes:
        if n.type in relevant_node_types:
            node_text = dedent('\n'.join(file_content.split('\n')[
                               n.start_point[0]:n.end_point[0]+1]))
            out.append(
                {'file': fp, 'line': n.start_point[0], 'text': node_text})
    return out


def get_repo_functions(root, supported_languages, relevant_node_types):
    functions = []
    for fp in [root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')]:
        if not os.path.isfile(fp):
            continue
        with open(fp, 'r') as f:
            lang = supported_languages.get(fp[fp.rfind('.'):])
            if lang:
                parser = Parser()
                parser.set_language(lang)
                file_content = f.read()
                tree = parser.parse(bytes(file_content, 'utf8'))
                all_nodes = list(traverse_tree(tree.root_node))
                functions.extend(extract_functions(
                    all_nodes, fp, file_content, relevant_node_types))
    return functions


def git_root():
    p = run(['git', 'rev-parse', '--show-toplevel'], capture_output=True)
    if p.returncode != 0:
        return None
    return p.stdout.decode('utf-8').strip()


def do_embed(model, root, language_lib_path, batch_size=32):
    functions = get_repo_functions(
        root,
        get_language_support(language_lib_path),
        ['function_definition', 'method_definition',
         'function_declaration', 'method_declaration']
    )

    print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
        len(functions), int(np.ceil(len(functions)/batch_size))))
    corpus_embeddings = model.encode(
        [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

    with gzip.open(root + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps(
            {'functions': functions, 'embeddings': corpus_embeddings}))


def search(query_embedding, corpus_embeddings, functions, k=5, file_extension=None):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k, sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.append((score, functions[idx]))
    return out


def do_query(model, root, query, file_extension=None, top_n=5):
    with gzip.open(root + '/' + '.embeddings', 'r') as f:
        query_embedding = model.encode(query, convert_to_tensor=True)
        dataset = pickle.loads(f.read())
        results = search(query_embedding, dataset.get(
            'embeddings'), dataset.get('functions'), k=top_n, file_extension=file_extension)
        return results


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def present_results(results, query, root):
    return inquirer.select(
        message='Go to result for query "{}"'.format(query),
        border=True,
        cycle=False,
        vi_mode=True,
        qmark='',
        amark='',
        pointer='👉  ',
        height='100%',
        raise_keyboard_interrupt=False,
        mandatory=False,
        choices=intersperse(
            [
                Choice((r[1]['file'], r[1]['line']+1),
                       name='{:.3f}'.format(r[0]) + ' ' + r[1]['file'].removeprefix(root + '/') + ':' + str(r[1]['line']) + '\n\n' + r[1]['text'].replace('\t', '  ')+'\n')
                for r in results],
            Separator()
        ) + [Choice(None, name='')],
        default=(results[0][1]['file'], results[0][1]['line']+1),
    ).execute()


def open_in_editor(file, line, editor):
    if editor == 'vim':
        os.system('vim +{} {}'.format(line, file))
    elif editor == 'vscode':
        os.system('code --goto {}:{}'.format(file, line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embed', action=argparse.BooleanOptionalAction)
    parser.add_argument('-q', '--query', type=str, required=False)
    parser.add_argument('-x', '--file-extension', type=str, required=False)
    parser.add_argument('-n', '--top-n', type=int, required=False, default=5)
    parser.add_argument('--language-lib-path', type=str, required=False)
    parser.add_argument(
        '--editor', choices=['vscode', 'vim'], default='vscode', required=False)
    args = parser.parse_args()

    open_ai_key = os.environ.get('OPENAI_API_KEY')
    if not open_ai_key:
        print('Please set the OPENAI_API_KEY environment variable e.g. export OPENAI_API_KEY=<my-key>')
        quit()

    openai.api_key = open_ai_key

    root = git_root()
    if not root:
        print('Not a git repository. Run this in a git repository')
        quit()

    # TODO: loading from disk takes almost a second, maybe run a background process?
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    if args.embed:
        do_embed(model, root, args.language_lib_path)
        quit()

    if args.query:
        if not os.path.isfile(root + '/' + '.embeddings'):
            print('Embeddings not found. Run with --embed')
            quit()
        results = do_query(model, root, args.query,
                           args.file_extension, args.top_n)

        file_path_with_line = present_results(
            results=results, query=args.query, root=root)
        if file_path_with_line is not None:
            open_in_editor(
                file_path_with_line[0], file_path_with_line[1], args.editor)
            quit()


if __name__ == "__main__":
    main()
