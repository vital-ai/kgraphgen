import json

from ai_haley_kg_domain.model.KGDocument import KGDocument
from ai_haley_kg_domain.model.KGInteraction import KGInteraction
from kgraphmemory.kginteraction_graph import KGInteractionGraph
from kgraphmemory.utils.uri_generator import URIGenerator
from vital_ai_vitalsigns.collection.graph_collection import GraphCollection
from vital_ai_vitalsigns.embedding.embedding_model import EmbeddingModel
from vital_ai_vitalsigns.vitalsigns import VitalSigns


def read_json_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs



def main():

    print('KGraphGen Dynamic Relationship Test')

    # load in graph via RDF

    # set dynamic entity of interest
    # in general all entities of a type like person
    # when building all entity to all entity
    # consider whether frame slot types are the same or different
    # for symmetric relationships they would be the same
    # for non-symmetric relationships they would be different
    # friend-of could be symmetric, whereas employed would be asymmetric

    # set dynamic relationship of interest

    # set target entity type of interest

    # via graph/vector query (sparql) find sentences that:
    # link to entity
    # link to an entity of target entity type
    # have relevant sentence via embedding vector
    # group sentences by target entity
    # take top N matches per target entity ranked by sentence relevance

    # for each of the found target entities
    # take sentences and use as context in prompt
    # ask model to confirm or deny relation

    # for confirmed cases
    # create frame of frame type relationship
    # frame includes entity link to source and target
    # add frame to graph

    # write updated graph out as RDF

    # initially use json data
    # create Graphs to use for vector search for:
    # entities, sentences, paragraphs

    # try friend-of and enemy-of sentences
    # links from resolved entities to sentence

    vs = VitalSigns()
    embedder = EmbeddingModel()
    vs.put_embedding_model(embedder.get_model_id(), embedder)

    interaction = KGInteraction()
    interaction.URI = URIGenerator.generate_uri()
    # print(interaction.to_json())

    docs = read_json_lines("../test_data_internal/hp_docs.2.jsonnl")

    clusters = read_json_lines("../test_output/hp_clusters.2.jsonnl")

    entity_list = read_json_lines("../test_output/hp_entity_list.2.jsonnl")

    sentence_graph = KGInteractionGraph(interaction)

    paragraph_graph  = KGInteractionGraph(interaction)

    doc_num = 0

    for doc in docs:

        doc_num += 1

        print(f"Loading Doc: {doc_num}")

        for paragraph in doc["paragraphs"]:

            paragraph_text = paragraph["text"]

            paragraph_doc = KGDocument()

            paragraph_doc.URI = URIGenerator.generate_uri()

            paragraph_doc.name = paragraph_text

            paragraph_graph.add_entity(paragraph_doc)

            sentences = paragraph.get("sentences", [])

            for sentence in sentences:

                sentence_doc = KGDocument()

                sentence_doc.URI = URIGenerator.generate_uri()

                sentence_text = sentence["text"]

                sentence_doc.name = sentence_text

                sentence_graph.add_entity(sentence_doc)

    print("Loaded data...")

    query = "friendship"

    results = paragraph_graph.graph.search(query, 'http://vital.ai/ontology/haley-ai-kg#KGDocument', 100)

    i = 0

    for result in results:

        i += 1

        query_sent = result.graph_object
        query_score = result.score

        print(f"PARAGRAPH ({i}) Score: {query_score}: {query_sent.name}")

    results = sentence_graph.graph.search(query, 'http://vital.ai/ontology/haley-ai-kg#KGDocument', 100)

    i = 0

    for result in results:

        i += 1

        query_sent = result.graph_object
        query_score = result.score

        print(f"SENTENCE ({i}) Score: {query_score}: {query_sent.name}")


if __name__ == "__main__":
    main()
