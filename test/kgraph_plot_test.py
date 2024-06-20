import json
import matplotlib.pyplot as plt
import networkx as nx


def read_json_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs


def wrap_label(label, width):
    """Wrap the label to a given width."""
    import textwrap
    return '\n'.join(textwrap.wrap(label, width))

def main():

    print('KGraphGen Plot Test')

    relations = read_json_lines("../test_output/hp_relation.2.jsonnl")

    nx_g = nx.Graph()

    node_map = {}

    # {"relation_id": 6, "relation_label": "Evil Behavior", "relation_summary": "Uncle Vernon punishes Harry by sending him to the cupboard without meals.", "entity_source": "Uncle Vernon", "entity_target": "Harry Potter", "paragraph_list": "99", "doc_num": 2}

    node_id = 0

    for r in relations:

        relation_label = r['relation_label']

        entity_source = r['entity_source']

        entity_target = r['entity_target']

        if node_map.get(entity_source) is None:
            node_map[entity_source] = node_id
            node_id += 1

        if node_map.get(entity_target) is None:
            node_map[entity_target] = node_id
            node_id += 1

    for k in node_map.keys():

        node_id = node_map[k]

        print(f"Node: {k}")

        nx_g.add_node(node_id, label=k)

    for r in relations:

        relation_label = r['relation_label']

        entity_source = r['entity_source']

        entity_target = r['entity_target']

        source_id = node_map[entity_source]

        target_id = node_map[entity_target]

        print(f"Relation: {source_id} --> {target_id} {relation_label}")

        nx_g.add_edge(source_id, target_id, label=relation_label)

    # pos = nx.spring_layout(nx_g, seed=42)

    pos = nx.kamada_kawai_layout(nx_g)

    # pos[0] = (-1, 1)

    fig, ax = plt.subplots(figsize=(12, 12))

    # plt.figure(figsize=(8, 6))
    # plt.xlim(-2, 2)
    # plt.ylim(-1.5, 1.5)

    nx.draw(nx_g, pos, node_color='skyblue', edge_color='gray', font_weight='bold', node_size=200, ax=ax)

    edge_labels = nx.get_edge_attributes(nx_g, 'label')

    # nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edge_labels, ax=ax)

    node_labels = nx.get_node_attributes(nx_g, 'label')

    nx.draw_networkx_labels(
        nx_g,
        pos,
        labels=node_labels,
        verticalalignment='bottom',
        horizontalalignment='center',
        font_size=8,
        ax=ax
        # bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'),
        # bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')
    )

    nx.draw_networkx_edge_labels(nx_g, pos, edge_labels=edge_labels, font_size=5, ax=ax)

    ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)
    ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.savefig("../test_output/hp_kgraph.2.png")

    plt.show()


if __name__ == "__main__":
    main()
