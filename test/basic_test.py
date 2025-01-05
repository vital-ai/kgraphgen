from typing import List
import json
import yaml
from ai_haley_kg_domain.model.Edge_hasInteractionKGDocument import Edge_hasInteractionKGDocument
from ai_haley_kg_domain.model.KGDocument import KGDocument
from ai_haley_kg_domain.model.KGEntity import KGEntity
from ai_haley_kg_domain.model.KGEntityType import KGEntityType
from ai_haley_kg_domain.model.KGInteraction import KGInteraction
from kgraphmemory.kginteraction_graph import KGInteractionGraph
from kgraphmemory.utils.uri_generator import URIGenerator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser, PydanticOutputFunctionsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate, \
    ChatPromptTemplate
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import load_tools, AgentType, initialize_agent, create_react_agent, AgentExecutor, \
    create_structured_chat_agent
from langchain.schema import (
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from vital_ai_vitalsigns.embedding.embedding_model import EmbeddingModel
from vital_ai_vitalsigns.vitalsigns import VitalSigns

from kgraphgen.extract.extract_client import ExtractClient


def load_config():
    with open("../agent_config.yaml", "r") as config_stream:
        try:
            return yaml.safe_load(config_stream)
        except yaml.YAMLError as exc:
            print("failed to load config file")


config = load_config()
openai_api_key = config['api_keys']['openai']


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetLocalTime(BaseModel):
    """Get the current time in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


# "gpt-4-turbo-preview"
# "gpt-4"
# "gpt-3.5-turbo"

# memory.save_context({"input": "[start]"}, {"output": "My name is Haley.  How can I help you?"})

# memory.save_context({"input": "My name is Marc."}, {"output": "Nice to meet you, Marc"})

# memory.save_context({"input": "I live in Brooklyn."}, {"output": "Brooklyn is awesome!"})

def get_data_file_content(file_path: str) -> str:
    with open(file_path, 'r') as file:
        file_contents = file.read()

    return file_contents


# get extract summary

# get extract named entities

# get extract named entity who-is/what-is
# use pronoun replacement to select all sentences where the
# entity was mentioned and use sumary model instead of qa?

# get extract related A & B
# use summary model instead with all sentences mentioning a and b?


class GenreSchema(BaseModel):
    """Genre classification."""
    genre_label: str = Field(..., description="The genre label")


def get_genre(kgraph: KGInteractionGraph, file_contents: str) -> str:

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model="gpt-4-turbo-preview")

    # openai_functions = [convert_pydantic_to_openai_function(GenreSchema)]

    # parser = PydanticOutputFunctionsParser(pydantic_schema=GenreSchema)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at understanding text and creating Knowledge Graphs.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # chain = prompt | llm.bind(functions=openai_functions) | parser

    chain = prompt | llm.with_structured_output(
        schema=GenreSchema,
        method="function_calling",
        include_raw=False,
    )

    genre_prompt = """
        Read the following text and provide the nature of the text in a single short label.
        Examples include, but are not limited to:
        business report
        medical profile
        non-fiction
        fiction/general
        fiction/scifi
        fiction/fantasy
        fiction/historical-drama
        """

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{genre_prompt}:\nText: {file_contents}"
        }
    )

    label = output.genre_label

    return label


# get named entities for genre

class GenreNamedEntitySchema(BaseModel):
    """Named Entity."""
    named_entity_type: str = Field(..., description="Label for the type of named entity")
    named_entity_type_description: str = Field(..., description="Description of the named entity type")


class GenreNamedEntityListSchema(BaseModel):
    """List of named entities."""
    named_entity_list: List[GenreNamedEntitySchema] = Field(..., description="List of named entities")


def get_genre_named_entities(kgraph: KGInteractionGraph, genre: str) -> list:

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model="gpt-4-turbo-preview")

    # openai_functions = [convert_pydantic_to_openai_function(GenreNamedEntityListSchema)]

    # parser = PydanticOutputFunctionsParser(pydantic_schema=GenreNamedEntityListSchema)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at understanding text and creating Knowledge Graphs.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # chain = prompt | llm.bind(functions=openai_functions) | parser

    chain = prompt | llm.with_structured_output(
        schema=GenreNamedEntityListSchema,
        method="function_calling",
        include_raw=False,
    )

    genre_entity_prompt = """
Given the following genre of a document, provide a list of the types of named entities
you would expect to find in the document.
For instance, in a business document you would expect:
businesses, employees, locations, products, revenue
Whereas in a novel you may expect:
Various kinds of people, locations, animals, groupings of people into teams, families
"""

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{genre_entity_prompt}:\nGenre: {genre}"
        }
    )

    entity_list = output.named_entity_list

    entity_map_list = []

    for e in entity_list:
        entity_map = {'entity_name': e.named_entity_type, 'entity_description': e.named_entity_type_description}
        entity_map_list.append(entity_map)

    return entity_map_list

# get relationships for genre


class GenreRelationshipParticipantSchema(BaseModel):
    """Relationship Participant."""
    relationship_participant_type: str = Field(..., description="Label for the role of the participant in the relationship")
    relationship_participant_description: str = Field(..., description="Description of the role of the participant in the relationship")


class GenreRelationshipSchema(BaseModel):
    """Relationship Type."""
    relationship_type: str = Field(..., description="Label for the type of relationship")
    relationship_type_description: str = Field(..., description="Description of the type of relationship")
    relationship_participant_list: List[GenreRelationshipParticipantSchema] = Field(..., description="List of the participants in the relationship")


class GenreRelationshipListSchema(BaseModel):
    """List of the relationship types."""
    relationship_list: List[GenreRelationshipSchema] = Field(..., description="List of the relationship types")


def get_genre_relationships(kgraph: KGInteractionGraph, genre: str, entity_list: list) -> list:

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name="gpt-4-turbo-preview")

    # openai_functions = [convert_pydantic_to_openai_function(GenreRelationshipListSchema)]

    # parser = PydanticOutputFunctionsParser(pydantic_schema=GenreRelationshipListSchema)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at understanding text and creating Knowledge Graphs.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # chain = prompt | llm.bind(functions=openai_functions) | parser

    chain = prompt | llm.with_structured_output(
        schema=GenreRelationshipListSchema,
        method="function_calling",
        include_raw=False,
    )

    markdown_table = "| Entity Type Name | Entity Type Description |\n"
    markdown_table += "|-------------|--------------------|\n"

    entity_count = len(entity_list)
    rel_count = entity_count * 2

    for entity in entity_list:
        markdown_table += f"| {entity['entity_name']} | {entity['entity_description']} |\n"

    genre_relationship_prompt = f"""
Given the following genre of a document, and a list of suggested entity types, 
provide a list of relationship types that you would typically see relating the entity types together.
A relationship may relate 2, 3, or more entity types together.
A participant has a participant type specifying the role in the relation.
Some relationships have participants of the same type, such as marriage which has two spouse participants.
But relationships typically have participants with different roles.
For instance, in a management relationship, one person has the role of manager, and the other has the role of being managed.
If you have a relationship type like "friend of" be sure to include its opposite "enemy of".
Cover all the suggested entity types at least, but suggest others if needed for your relationship types.
You should provide at least {rel_count} relationship types.
Examples:
If the entity type was business and the genre was business document, an expected
relationship could be a company acquisition which would relate a seller business and a buyer business.
If the entity was a character in a novel, then marriage would be a common relationship which would involve
two people as spouses.
If the entity was a person in a non fiction story, parent-of would be a common relationship, which would have one participant have the role of child
and the other participant have the role of parent.
If the entity was a spaceship in a sci-fi novel, then a relationship could be commander-of which would relate the 
spaceship and the captain in charge of the ship.
"""

    chain_input = f"{genre_relationship_prompt}:\nGenre: {genre}\nNamed Entity Types:\n{markdown_table}"

    print(f"Chain Input:\n{chain_input}")

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": chain_input
        }
    )

    relationship_list = output.relationship_list

    relationship_map_list = []

    for r in relationship_list:

        participant_list = []

        for p in r.relationship_participant_list:

            participant_map = {
                'participant_type': p.relationship_participant_type,
                'participant_description': p.relationship_participant_description
            }

            participant_list.append(participant_map)

        relationship_map = {
            'relationship_type': r.relationship_type,
            'relationship_type_description': r.relationship_type_description,
            'relationship_participant_list': participant_list
        }

        relationship_map_list.append(relationship_map)

    return relationship_map_list


# get named entities and description from text


class NamedEntitySchema(BaseModel):
    """Named Entity."""
    named_entity: str = Field(..., description="Name of named entity")
    named_entity_type: str = Field(..., description="Label for the type of named entity")
    named_entity_type_description: str = Field(..., description="Description of the named entity type")


class NamedEntityListSchema(BaseModel):
    """List of named entities."""
    named_entity_list: List[NamedEntitySchema] = Field(..., description="List of named entities")


def get_named_entities_text(kgraph: KGInteractionGraph, text: str, genre: str, genre_entity_list: list) -> list:

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name="gpt-4-turbo-preview")

    # openai_functions = [convert_pydantic_to_openai_function(NamedEntityListSchema)]

    # parser = PydanticOutputFunctionsParser(pydantic_schema=NamedEntityListSchema)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at understanding text and creating Knowledge Graphs.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # chain = prompt | llm.bind(functions=openai_functions) | parser

    chain = prompt | llm.with_structured_output(
        schema=NamedEntityListSchema,
        method="function_calling",
        include_raw=False,
    )

    markdown_table = "| Entity Type Name | Entity Type Description |\n"
    markdown_table += "|-------------|--------------------|\n"

    entity_count = len(genre_entity_list)

    for entity in genre_entity_list:
        markdown_table += f"| {entity['entity_name']} | {entity['entity_description']} |\n"

    named_entity_prompt = f"""
Given the following genre of a document, and a list of suggested entity types, 
provide a list of named entity you extract from the document.
You should be very thorough and extract every named entity you find.
You should use the entity types when it is a match, but otherwise add a new
entity type for the named entity.
Examples:
If the genre was business document, you should extract all named entities such as
businesses, people, locations, products, monetary values, dates of events, and so on.
If the genre is a navel, then you should extract people of various kinds, places, animals, and so on.
"""

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{named_entity_prompt}:\nGenre: {genre}\nNamed Entity Types:\n{markdown_table}\n:Test:\n{text}"
        }
    )

    named_entity_list = output.named_entity_list

    entity_map_list = []

    for e in named_entity_list:

        entity_map = {
            'named_entity': e.named_entity,
            'named_entity_type': e.named_entity_type,
            'named_entity_type_description': e.named_entity_type_description
        }

        entity_map_list.append(entity_map)

    return entity_map_list

# get relationships and description from text


class RelationshipParticipantSchema(BaseModel):
    """Relationship Participant."""
    participant_name: str = Field(..., description="The name of the relationship participant")
    participant_type: str = Field(..., description="Label for the role of the relationship participant")
    participant_description: str = Field(..., description="Description of the role of relationship participant")


class RelationshipSchema(BaseModel):
    """Relationship."""
    relationship_count: int = Field(..., description="The integer ordinal number of this relationship")
    relationship_type: str = Field(..., description="Label for the type of relationship")
    relationship_type_description: str = Field(..., description="Description of the type of relationship")
    relationship_participant_list: List[RelationshipParticipantSchema] = Field(..., description="List of the participants in the relationship")


class RelationshipListSchema(BaseModel):
    """List of the relationships."""
    relationship_list: List[RelationshipSchema] = Field(..., description="List of the relationships")


def get_relationships_text(kgraph: KGInteractionGraph, text, entity_list: list, genre, genre_entity_list, genre_relationship_list: list) -> list:

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )
    # setting max tokens to try and not get json errors
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        max_tokens=3_000,
        temperature=0,
        model_name="gpt-4-turbo-preview")

    # openai_functions = [convert_pydantic_to_openai_function(RelationshipListSchema)]

    # parser = PydanticOutputFunctionsParser(pydantic_schema=RelationshipListSchema)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at understanding text and creating Knowledge Graphs.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    # chain = prompt | llm.bind(functions=openai_functions) | parser

    chain = prompt | llm.with_structured_output(
        schema=RelationshipListSchema,
        method="function_calling",
        include_raw=False,
    )

    named_entity_count = len(entity_list)

    relationship_target_count = named_entity_count

    # named entity markdown
    named_entity_markdown_table = "| Named Entity Name | Named Entity Type | Named Entity Description |\n"
    named_entity_markdown_table += "|-------------|----------|--------------------|\n"

    for entity in entity_list:
        named_entity_markdown_table += f"| {entity['named_entity']} | {entity['named_entity_type']} | {entity['named_entity_type_description']} |\n"

    # genre entity markdown
    entity_markdown_table = "| Entity Type Name | Entity Type Description |\n"
    entity_markdown_table += "|-------------|--------------------|\n"

    for entity in genre_entity_list:
        entity_markdown_table += f"| {entity['entity_name']} | {entity['entity_description']} |\n"

    # genre relationship markdown
    relationship_markdown_table = ""

    for relationship in genre_relationship_list:
        relationship_markdown_table += f"### Relationship Type: {relationship['relationship_type']}\n"
        relationship_markdown_table += f"**Description**: {relationship['relationship_type_description']}\n\n"

        # Table header for the participants
        relationship_markdown_table += "| Participant Type | Participant Description |\n"
        relationship_markdown_table += "|------------------|-------------------------|\n"

        # Rows for each participant
        for participant in relationship['relationship_participant_list']:
            relationship_markdown_table += f"| {participant['participant_type']} | {participant['participant_description']} |\n"

        # Add a newline for spacing between sections
        relationship_markdown_table += "\n"

    relationship_prompt = f"""
Given the following genre of a document, a list of named entities found in the document, 
a list of suggested entity types, and a list of suggested relationship types, 
provide a list of relationships you extract from the document.
A participant has a participant type specifying the role in the relationship.
Some relationships have participants of the same type, such as marriage which has two spouse participants.
But relationships typically have participants with different roles.
For instance, in a manager role, one person has the role of manager, and the other has the role of being managed.
You should be thorough and extract EVERY relationship.
Each named entity should participate in at least one relationship.
Based on the Named Entity Count of {named_entity_count} you should extract at least {relationship_target_count} relationships.
Include the ordinal number on the relationship extracted so you can be sure to keep track of how many are on the list so
you can be sure to make at least {relationship_target_count} relationships.
You should use the named entities from the list.
You should use the suggested named entity types and suggested relationship types if appropriate,
but you can define new entity types and new relationship types if needed.
Examples:
If the entity was business and the genre was business document, an expected
relationship could be a company acquisition which would relate a seller business and a buyer business.
If the entity was a character in a novel, then marriage would be a common relationship which would involve
two people as spouses.
If the entity was a spaceship in a sci-fi novel, then a relationship could be commander-of which would relate the 
spaceship and the captain in charge of the ship.
"""

    chain_input = f"{relationship_prompt}:\nGenre: {genre}\nNamed Entities:\n{named_entity_markdown_table}\nNamed Entity Types:\n{entity_markdown_table}\nRelationship Types:\n{relationship_markdown_table}\nText:\n{text}"

    print(f"Relationship Prompt:\n{chain_input}")

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": chain_input
        }
    )

    relationship_list = output.relationship_list

    relationship_map_list = []

    for r in relationship_list:

        participant_list = []

        for p in r.relationship_participant_list:

            participant_map = {
                'participant_name': p.participant_name,
                'participant_type': p.participant_type,
                'participant_description': p.participant_description
            }

            participant_list.append(participant_map)

        relationship_map = {
            'relationship_count': r.relationship_count,
            'relationship_type': r.relationship_type,
            'relationship_type_description': r.relationship_type_description,
            'relationship_participant_list': participant_list
        }

        relationship_map_list.append(relationship_map)

    return relationship_map_list


# insert entities and frames into kgraphmemory
# use frames for relationships with slot type for the
# type of participation in relationship


def insert_entities_relationships(kgraph: KGInteractionGraph, entity_list: list, relationship_list: list) -> list:
    pass


def ask_question_kgraph(kgraph: KGInteractionGraph, question: str) -> list:
    pass

# ask function call question from LLM
# LLM fills in function call including entity and relationship types
# query kgraph memory for data
# provide results into LLM for answer

# visualize kgraph memory


def visualize_kgraph(kgraph: KGInteractionGraph):
    pass


def main():

    print('KGraphGen Basic Test')

    vs = VitalSigns()
    embedder = EmbeddingModel()
    vs.put_embedding_model(embedder.get_model_id(), embedder)

    ################################################################
    # initialize extract client
    client = ExtractClient("localhost", 6111)

    ################################################################
    # initialize graph and interaction
    interaction = KGInteraction()
    interaction.URI = URIGenerator.generate_uri()

    # print(interaction.to_json())

    kgraph = KGInteractionGraph(interaction)

    # save kgraph memory as rdf file after each step

    ################################################################
    # load data file

    file_path = '../test_data/hp_plot.txt'

    file_contents = get_data_file_content(file_path)

    # print(file_contents)

    doc = KGDocument()
    doc.URI = URIGenerator.generate_uri()

    doc.kGDocumentContent = file_contents

    doc_edge = Edge_hasInteractionKGDocument()
    doc_edge.URI = URIGenerator.generate_uri()

    # adjust serialization for URIs
    doc_edge.edgeSource = str(interaction.URI)
    doc_edge.edgeDestination = str(doc.URI)

    graph_list = [doc, doc_edge]

    for g in graph_list:
        kgraph.graph.append(g)

    for g in kgraph.graph:
        # fix serialization issue
        print(g.to_json())
        # print(g)

    ################################################################
    # extract from document

    data = {
        "task": {
            "steps": [
                "extract"
            ]
        },
        "data": {
            "document_list": [
                {
                    "doc_id": str(doc.URI),
                    "doc_content": str(doc.kGDocumentContent)
                }
            ]
        }
    }

    results = client.extract(data)

    print(results)


    ################################################################
    # determine genre of text
    genre = get_genre(kgraph, file_contents)
    print(f"Genre: {genre}")

    doc.kGDocumentSummary = genre

    ################################################################
    # get named entities for genre
    genre_entity_list = get_genre_named_entities(kgraph, genre)
    print(f"Genre Entity List: {genre_entity_list}")

    for e in genre_entity_list:
        pretty_json = json.dumps(e, indent=4)

        # print(pretty_json)

        entity_type_name = e.get('entity_name')

        entity_description = e.get('entity_description')

        kg_entity_type = KGEntityType()
        kg_entity_type.URI = URIGenerator.generate_uri()
        kg_entity_type.name = entity_type_name
        kg_entity_type.kGraphDescription = entity_description

        print(kg_entity_type.to_json())

        kgraph.graph.append(kg_entity_type)

        # print(e)

    exit(0)

    # get relationships for genre
    genre_relationship_list = get_genre_relationships(kgraph, genre, genre_entity_list)
    print(f"Genre Relationship List: {genre_relationship_list}")

    for r in genre_relationship_list:
        pretty_json = json.dumps(r, indent=4)
        print(pretty_json)

    ################################################################
    # get named entities and description from text
    named_entity_list = get_named_entities_text(kgraph, file_contents, genre, genre_entity_list)
    print(f"Named Entity List: {named_entity_list}")

    for e in named_entity_list:
        pretty_json = json.dumps(e, indent=4)
        print(pretty_json)

    ################################################################
    # get relationships and description from text
    relationship_list = get_relationships_text(kgraph, file_contents, named_entity_list, genre, genre_entity_list, genre_relationship_list)
    print(f"Relationship List: {relationship_list}")

    for r in relationship_list:
        pretty_json = json.dumps(r, indent=4)
        print(pretty_json)

    ################################################################
    # insert entities and frames into kgraphmemory
    # use frames for relationships with slot type for the
    # type of participation in relationship

    ################################################################
    # ask function call question from LLM
    # LLM fills in function call including entity and relationship types
    # query kgraph memory for data
    # provide results into LLM for answer

    ################################################################
    # visualize kgraph memory


if __name__ == "__main__":
    main()
