
from sentence_transformers import SentenceTransformer
from tree_sitter import Tree
from tree_sitter_languages import get_parser
from textwrap import dedent
import os
import numpy as np
import pickle
import gzip


def supported_file_extensions():
    return {
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.java': 'java',
        '.vue': 'vue',
        '.svelte': 'svelte',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.py': 'python'
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


def get_repo_functions(root, supported_file_extensions, relevant_node_types):
    functions = []
    for fp in [root + '/' + f for f in os.popen('git -C {} ls-files'.format(root)).read().split('\n')]:
        if not os.path.isfile(fp):
            continue
        with open(fp, 'r') as f:
            lang = supported_file_extensions.get(fp[fp.rfind('.'):])
            if lang:
                parser = get_parser(lang)
                file_content = f.read()
                tree = parser.parse(bytes(file_content, 'utf8'))
                all_nodes = list(traverse_tree(tree.root_node))
                functions.extend(extract_functions(
                    all_nodes, fp, file_content, relevant_node_types))
    return functions


def do_embed(args):
    model = SentenceTransformer(args.model_name_or_path)
    nodes_to_extract = ['function_definition', 'method_definition',
                        'function_declaration', 'method_declaration']
    functions = get_repo_functions(
        args.path_to_repo, supported_file_extensions(), nodes_to_extract)

    print('Embedding {} functions in {} batches. This is done once and cached in .embeddings'.format(
        len(functions), int(np.ceil(len(functions)/args.batch_size))))
    corpus_embeddings = model.encode(
        [f['text'] for f in functions], convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)

    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'w') as f:
        f.write(pickle.dumps(
            {'functions': functions, 'embeddings': corpus_embeddings}))
