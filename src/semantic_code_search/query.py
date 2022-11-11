import os
from sentence_transformers import SentenceTransformer, util
import gzip
import pickle
import torch
from . import prompt


def search(query_embedding, corpus_embeddings, functions, k=5, file_extension=None):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k, sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.append((score, functions[idx]))
    return out


def foo(model, root, query, file_extension=None, top_n=5):
    with gzip.open(root + '/' + '.embeddings', 'r') as f:
        query_embedding = model.encode(query, convert_to_tensor=True)
        dataset = pickle.loads(f.read())
        results = search(query_embedding, dataset.get(
            'embeddings'), dataset.get('functions'), k=top_n, file_extension=file_extension)
        return results


def do_query(args):
    if not args.query_text:
        print('provide a query')
        # todo: add a prompt here as a fallback
        quit()

    if not os.path.isfile(args.path_to_repo + '/' + '.embeddings'):
        print('Embeddings not found in {}. Run with --embed'.format(args.path_to_repo))
        quit()

    model = SentenceTransformer(args.model_name_or_path)

    results = foo(model, args.path_to_repo, args.query_text,
                  args.file_extension, args.n_results)

    file_path_with_line = prompt.present_results(
        results=results, query=args.query_text, root=args.path_to_repo)
    if file_path_with_line is not None:
        prompt.open_in_editor(
            file_path_with_line[0], file_path_with_line[1], args.editor)
        quit()
