from InquirerPy.separator import Separator
from InquirerPy import get_style
from InquirerPy.base.control import Choice
from InquirerPy import inquirer
import gzip
from subprocess import run
from tqdm import tqdm
import openai
import argparse
import pickle
from sklearn.neighbors import KDTree
import time

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


pl_model = 'code-search-babbage-code-001'
nl_model = 'code-search-babbage-text-001'


def get_embedding(text, isCode=True):
    if not isCode:
        text = text.replace("\n", " ")
    model = pl_model if isCode else nl_model
    try:
        response = openai.Embedding.create(input=[text], model=model)
    except openai.error.APIError as e:
        print(e)
        time.sleep(10)
        response = openai.Embedding.create(
            input=[text], model=model)  # try one more time
    return response['data'][0]['embedding']


def do_embed(root, language_lib_path):
    functions = get_repo_functions(
        root,
        get_language_support(language_lib_path),
        ['function_definition', 'method_definition',
         'function_declaration', 'method_declaration']
    )

    for f in functions:
        if len(f['text']) > 3000:  # truncate for the open ai max token limit
            f['text'] = f['text'][:3000]
    embeddings = []
    print('Embedding {} functions. This is done once and cached in .embeddings'.format(
        len(functions)))
    for f in tqdm(functions):
        embeddings.append(get_embedding(f['text'], isCode=True))
    kdt = KDTree(np.array(embeddings), leaf_size=30, metric='euclidean')
    with gzip.open(root + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps({'functions': functions, 'kdt': kdt}))


def search(query_embedding, dataset, n=5):
    (distances, nbrs) = dataset.get('kdt').query(
        [query_embedding], k=n, return_distance=True, sort_results=True)
    result = []
    for idx, nbr in enumerate(nbrs[0]):
        result.append((distances[0][idx], dataset.get('functions')[nbr]))
    result.reverse()
    return result


def do_query(root, query, file_extension=None, top_n=5):
    with gzip.open(root + '/' + '.embeddings', 'r') as f:
        query_embedding = get_embedding(query, isCode=False)
        dataset = pickle.loads(f.read())
        result = search(query_embedding, dataset, n=top_n)
        if file_extension:
            result = [r for r in result if r[1]
                      ['file'].endswith(file_extension)]
        return result


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

    if args.embed:
        do_embed(root, args.language_lib_path)
        quit()

    if args.query:
        if not os.path.isfile(root + '/' + '.embeddings'):
            print('Embeddings not found. Run with --embed')
            quit()
        results = do_query(root, args.query, args.file_extension, args.top_n)
        file_path_with_line = present_results(
            results=results, query=args.query, root=root)
        if file_path_with_line is not None:
            open_in_editor(
                file_path_with_line[0], file_path_with_line[1], args.editor)
            quit()


if __name__ == "__main__":
    main()
