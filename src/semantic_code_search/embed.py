
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser, Tree
from textwrap import dedent
import os
import numpy as np
import pickle
import gzip


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


def do_embed(args):
    model = SentenceTransformer(args.model_name_or_path)
    languages = get_language_support(args.language_lib_path)
    nodes_to_extract = ['function_definition', 'method_definition',
                        'function_declaration', 'method_declaration']
    functions = get_repo_functions(
        args.path_to_repo, languages, nodes_to_extract)

    print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
        len(functions), int(np.ceil(len(functions)/args.batch_size))))
    corpus_embeddings = model.encode(
        [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)

    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps(
            {'functions': functions, 'embeddings': corpus_embeddings}))
