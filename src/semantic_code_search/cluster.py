import gzip
import os
import pickle
from semantic_code_search.embed import do_embed
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from textwrap import indent


def _get_clusters(dataset, distance_threshold):
    embeddings = dataset.get('embeddings')
    # Normalize the embeddings to unit length
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dataset['embeddings'] = embeddings

    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_distances=True,
    )
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_distances = clustering_model.distances_
    cluster_children = clustering_model.children_

    clustered_functions = {}
    for idx, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_functions:
            clustered_functions[cluster_id] = []

        ds_entry = dataset.get('functions')[idx]
        ds_entry['idx'] = idx

        clustered_functions[cluster_id].append(ds_entry)

    # filter out clusters with only one function
    clusters = []
    for cluster_id, functions in clustered_functions.items():
        if len(functions) > 1:
            fx_idx = functions[0].get('idx')
            distances = []
            for f in functions[1:]:
                f_idx = f.get('idx')
                for i, cc in enumerate(cluster_children):
                    if cc.tolist() == [fx_idx, f_idx]:
                        distances.append(cluster_distances[i])
            avg_distance = sum(distances) / \
                len(distances) if len(distances) > 0 else 0
            clusters.append(
                {'avg_distance': avg_distance, 'functions': functions})

    return clusters


def do_cluster(args, model):
    if not os.path.isfile(args.path_to_repo + '/' + '.embeddings'):
        print('Embeddings not found in {}. Generating embeddings now.'.format(
            args.path_to_repo))
        do_embed(args, model)

    with gzip.open(args.path_to_repo + '/' + '.embeddings', 'r') as f:
        dataset = pickle.loads(f.read())
        if dataset.get('model_name') != args.model_name_or_path:
            print('Model name mismatch. Regenerating embeddings.')
            dataset = do_embed(args, model)
        clusters = _get_clusters(dataset, args.cluster_max_distance)

        filtered_clusters = []
        for c in (clusters):
            if args.cluster_ignore_identincal and c.get('avg_distance') == 0:
                continue
            if any([len(f.get('text').split('\n')) <= args.cluster_min_lines for f in c.get('functions')]):
                continue
            if len(c.get('functions')) < args.cluster_min_cluster_size:
                continue
            filtered_clusters.append(c)

        for i, c in enumerate(filtered_clusters):
            print('Cluster #{}: avg_distance: {:.3} ================================================\n'.format(
                i, c.get('avg_distance')))
            # print('avg_distance:', c.get('avg_distance'))
            for f in c.get('functions'):
                print(indent(f.get('file'), '    ') + ':' + str(f.get('line')))
                print(indent(f.get('text'), '    ') + '\n')
