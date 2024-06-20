import random
from typing import List
import csv
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


def load_config():
    with open("../agent_config.yaml", "r") as config_stream:
        try:
            return yaml.safe_load(config_stream)
        except yaml.YAMLError as exc:
            print("failed to load config file")


config = load_config()
openai_api_key = config['api_keys']['openai']


def read_json_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs


class EntitySchema(BaseModel):
    """Entity."""
    entity_id: int = Field(..., description="The integer ordinal number of this entity")
    entity_label: str = Field(..., description="Label for the entity")
    entity_description: str = Field(..., description="Description of the entity")
    entity_cluster_list: List[int] = Field(..., description="List of the cluster identifier mapping to this entity")


class EntityListSchema(BaseModel):
    """List of the resolved entities."""
    entity_list: List[EntitySchema] = Field(..., description="List of the resolved entities")


def select_random_items(items):
    num_items = random.randint(1, min(5, len(items)))
    return random.sample(items, num_items)


def resolve_named_entities(docs: list, clusters: list):

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model="gpt-4o")

    # "gpt-4-turbo-preview"

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at resolving named entities in documents to help form a Knowledge Graph.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    chain = prompt | llm.with_structured_output(
        schema=EntityListSchema,
        method="function_calling",
        include_raw=False,
    )

    cluster_description = ""

    for cluster in clusters:

        name_list = cluster['name_list']
        name_list_str = ", ".join(name_list)
        internal_cluster_id = cluster['internal_cluster_id']
        cluster_id = cluster['cluster_id']

        data = cluster['data']
        sample_data = select_random_items(data)

        # cluster_description += "--------------------------\n"
        cluster_description += f"Cluster ID: {cluster_id}\n"
        cluster_description += f"Name List: {name_list_str}\n"

        for d in sample_data:
            cluster_description += f"Sample usage: {d['sentence']}\n"

        cluster_description += "--------------------------\n"

    print(f"Prompt:\n{cluster_description}")

    named_entity_prompt = f"""
    You are an expert in cleaning up data extracted from documents.
    You are cleaning up data collected in a text clustering process.
    You must only use information in the provided information below.
    Even if the names sound like a famous entity you know, only use the knowledge provided here.
    You are resolving the messy labels into a good list of unique Person Named Entities.
    Your task is to: 
        Review the following list of named entities and sample sentences.
        Eliminate those that are not names of people.
        Consolidate the people into a unique Person list, removing duplicates.
        Make sure all your entities are unique people with no duplicates whatsoever.
        Choose a good name for each person.
        Provide a description based on anything you learn in the sample usages.
        If a person has a nickname or alternate name, put it into the description.
        Provide the list of cluster identifiers you map to each of your results.
    Text Clusters:    
    {cluster_description}
    """

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{named_entity_prompt}"
        }
    )

    entity_list = output.entity_list

    entity_map = {'entity_list': entity_list}

    return entity_map


def main():

    print('KGraphGen Resolve Named Entity Test')

    docs = read_json_lines("../test_data_internal/hp_docs.2.jsonnl")

    clusters = read_json_lines("../test_output/hp_clusters.2.jsonnl")

    # pass entities and cluster context to resolver

    resolved_entities = resolve_named_entities(docs, clusters)

    # write resolved entities

    print(resolved_entities)

    entity_list = resolved_entities['entity_list']

    entity_map_list = []

    for entity in entity_list:

        cluster_mapping_str = ", ".join(map(str, entity.entity_cluster_list))

        entity_map = {
            "entity_id": entity.entity_id,
            "entity_label": entity.entity_label,
            "entity_cluster_list": entity.entity_cluster_list,
            "entity_description": entity.entity_description
        }

        entity_map_list.append(entity_map)

        print(f"Entity ID: {entity.entity_id} Label: {entity.entity_label}")
        print(f"Entity Description: {entity.entity_description}")
        print(f"Cluster Mapping: {cluster_mapping_str}")
        print()

    with open("../test_output/hp_entity_list.2.jsonnl", "w", encoding="utf-8") as f:
        for entity_dict in entity_map_list:
            f.write(json.dumps(entity_dict) + "\n")

    for entity in entity_map_list:
        entity['entity_cluster_list'] = ', '.join(map(str, entity['entity_cluster_list']))

    with open('../test_output/hp_entity_list.2.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=entity_map_list[0].keys())
        writer.writeheader()
        for entity in entity_map_list:
            writer.writerow(entity)


if __name__ == "__main__":
    main()

