import json
import matplotlib.pyplot as plt
import networkx as nx
import csv


def main():

    print('KGraphGen Plot CSVTest')

    filename = '../test_output/openai_worksfor_extract.csv'

    G = nx.Graph()

    nodes = set()

    # Read the CSV file into a list
    rows = []
    with open(filename, mode='r', newline='', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        source_name = row["Source Name"]
        source_type = row["Source Entity Type"]
        target_name = row["Target Name"]
        target_type = row["Target Entity Type"]

        # Add nodes for source and target entities
        nodes.add((source_name, source_type))
        nodes.add((target_name, target_type))

    # Add nodes to the graph
    for name, entity_type in nodes:
        G.add_node(name, entity_type=entity_type)

    # Second iteration to create edges
    for row in rows:
        source_name = row["Source Name"]
        target_name = row["Target Name"]
        relation_type = row["Relation Type"]

        # Add edges between source and target entities
        G.add_edge(source_name, target_name, relation=relation_type)

    # Draw the graph
    # pos = nx.kamada_kawai_layout(G)
    # plt.figure(figsize=(15, 10))

    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(20, 15))

    # Draw nodes with different colors based on entity type
    colors = ['lightblue' if G.nodes[node]['entity_type'] == 'PERSON' else 'lightgreen' for node in G.nodes()]
    # nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10, font_weight='bold',edge_color='black')

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_size=8, font_weight='bold',
            edge_color='gray')

    edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('OpenAI WorksFor People and Organizations')

    plt.subplots_adjust(top=0.95)

    plt.savefig('../test_output/openai_worksfor_extract.png', bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
