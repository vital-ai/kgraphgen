import json
import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


def read_json_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs


def find_data(docs):

    data = []

    entity_id = 1

    for doc in docs:
        text = doc["text"]

        paragraph_num = 0

        for paragraph in doc["paragraphs"]:

            paragraph_num += 1

            paragraph_text = paragraph["text"]
            paragraph_start = paragraph["start"]
            paragraph_end = paragraph["end"]
            sentences = paragraph.get("sentences", [])
            entities = paragraph.get("entities", [])

            for entity in entities:

                entity_label = entity.get("label", None)

                if entity_label == 'PERSON':

                    entity_id += 1

                    entity_text = entity["text"]
                    entity_start = entity["start"]
                    entity_end = entity["end"]

                    # Find the sentence containing the entity
                    sentence_containing_entity = None

                    sentence_num = 0

                    for sentence in sentences:

                        sentence_num += 1

                        sentence_text = sentence['text']

                        if sentence["start"] <= entity_start and sentence["end"] >= entity_end:
                            sentence_containing_entity = sentence
                            break

                    print(f"Entity: {entity_text} ({entity_label})")

                    # print(f"  Paragraph: {paragraph_text}")
                    # print(f"  Sentence: {sentence_containing_entity['text'] if sentence_containing_entity else 'Not found'}")

                    print()

                    entity_map = {
                        "id": entity_id,
                        "sentence_num": sentence_num,
                        "paragraph_num": paragraph_num,
                        "name": entity_text,
                        "sentence": sentence_text,
                        "paragraph": paragraph_text,
                        "context": paragraph_text
                    }

                    data.append(entity_map)

    return data


def print_cluster_hierarchy(data, tree, clusters):
    # Convert tree to a pandas DataFrame
    tree_df = tree.to_pandas()
    # Select only the clusters (exclude single points)
    cluster_tree = tree_df[tree_df.child_size > 1]
    # Sort by the parent then by lambda value in descending order
    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    def print_node(cluster_id, indent=0):
        # Find children of the current cluster
        children = cluster_tree[cluster_tree.parent == cluster_id]
        if not children.empty:
            for _, row in children.iterrows():

                print(' ' * indent + f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

                internal_cluster_id = int(row.child)

                clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

                unique_names = set()

                for cluster_id, items in clusters_data.items():
                    # print(f"Internal Cluster ID: {internal_cluster_id} Cluster ID {cluster_id} contains: {items}")
                    for item in items:
                        name = item['name']
                        unique_names.add(name)

                print(', '.join(unique_names))

                # Recurse to print each child
                print_node(row.child, indent + 4)

    # Find the roots
    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]
    for _, row in roots.iterrows():
        print(f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

        print_node(row.child, 4)


def find_final_clusters_in_top_cluster(data, tree, top_cluster_id, clusters):
    tree_df = tree.to_pandas()

    # Function to recursively collect all points under a given cluster node
    def collect_points(cluster_id):
        points = set()
        children = tree_df[tree_df['parent'] == cluster_id]
        for _, child in children.iterrows():
            if child['child_size'] == 1:  # It's a leaf node
                points.add(int(child['child']))
            else:
                points.update(collect_points(int(child['child'])))
        return points

    # Collect all points starting from the top_cluster_id
    points_in_top_cluster = collect_points(top_cluster_id)

    # Map these points to their final cluster IDs
    final_cluster_mapping = {point: clusters[point] for point in points_in_top_cluster if clusters[point] != -1}

    # Retrieve the corresponding data items, mapping them to their final clusters
    final_clusters_data = {cluster_id: [] for cluster_id in set(final_cluster_mapping.values())}
    for point, cluster_id in final_cluster_mapping.items():
        final_clusters_data[cluster_id].append(data[point])

    return final_clusters_data


def filter_cluster_hierarchy(data, tree, clusters):

    filtered_nodes = []

    tree_df = tree.to_pandas()
    cluster_tree = tree_df[tree_df.child_size > 1]
    cluster_tree = cluster_tree.sort_values(by=['parent', 'lambda_val'], ascending=[True, False])

    def filter_node(cluster_id):

        children = cluster_tree[cluster_tree.parent == cluster_id]

        if not children.empty:
            for _, row in children.iterrows():

                # print(f'Cluster {int(row.child)}, Size: {int(row.child_size)}')

                internal_cluster_id = int(row.child)

                clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

                unique_names = set()

                for cluster_id, items in clusters_data.items():
                    # print(f"Internal Cluster ID: {internal_cluster_id} Cluster ID {cluster_id} contains: {items}")
                    for item in items:
                        name = item['name']
                        unique_names.add(name)

                if len(unique_names) <= 4:
                    filter_map = {"names": unique_names, "internal_cluster_id": internal_cluster_id, "cluster_id": cluster_id}
                    filtered_nodes.append(filter_map)
                else:
                    filter_node(row.child)

    roots = cluster_tree[~cluster_tree.parent.isin(cluster_tree.child)]

    for _, row in roots.iterrows():

        filter_node(row.child)

    return filtered_nodes


def main():

    print('KGraphGen Cluster Entity Test')

    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    docs = read_json_lines("../test_data_internal/hp_docs.2.jsonnl")

    data = find_data(docs)

    name_embeddings = model.encode([d['name'] for d in data])

    context_embeddings = model.encode([d['context'] for d in data])

    features = np.hstack((name_embeddings, context_embeddings))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)

    clusters = clusterer.fit_predict(features)

    for item, cluster in zip(data, clusters):
        print(f"ID: {item['id']}, Name: {item['name']}, Cluster: {cluster}")
        item['cluster_id'] = cluster

    tree = clusterer.condensed_tree_

    print_cluster_hierarchy(data, tree, clusters)

    filtered_nodes = filter_cluster_hierarchy(data, tree, clusters)

    data_length = len(data)

    print(f"Data Length: {data_length}")

    for fn in filtered_nodes:
        print(fn)

    print(f"Filtered Node: {len(filtered_nodes)}")

    cluster_list = []

    for fn in filtered_nodes:

        names = list(fn['names'])

        internal_cluster_id = int(fn['internal_cluster_id'])

        cluster_id = int(fn['cluster_id'])

        clusters_data = find_final_clusters_in_top_cluster(data, tree, internal_cluster_id, clusters)

        cluster = {
            "name_list": names,
            "internal_cluster_id": internal_cluster_id,
            "cluster_id": cluster_id
        }

        for cluster_id, items in clusters_data.items():
            cluster["data"] = items

            for item in items:
                print(item)

        cluster_list.append(cluster)

    with open("../test_output/hp_clusters.2.jsonnl", "w", encoding="utf-8") as f:
        for cluster_dict in cluster_list:
            f.write(json.dumps(convert_types(cluster_dict)) + "\n")


def convert_types(data):
    if isinstance(data, dict):
        return {k: convert_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_types(item) for item in data]
    elif isinstance(data, set):
        return list(data)
    elif isinstance(data, (np.integer, pd.Int64Dtype().type)):
        return int(data)
    else:
        return data


if __name__ == "__main__":
    main()
