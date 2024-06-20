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
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
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


def summarize_doc(doc):
    pass


class RelationSchema(BaseModel):
    """Relation between two Entities."""
    relation_id: int = Field(..., description="The integer ordinal number of this relation")
    relation_label: str = Field(..., description="Label for the type of relation")
    relation_summary: str = Field(..., description="Short Summary of this specific relation between the entities with support from the text")
    entity_source: str = Field(..., description="Name of entity that is the source of the relation")
    entity_target: str = Field(..., description="Name of entity that is the target of the relation")
    # symmetric: bool = Field(..., description="Whether the relation is symmetric")
    paragraph_list: List[int] = Field(..., description="List of paragraph ids that support this relation")


class RelationListSchema(BaseModel):
    """List of the relations between entities."""
    relation_list: List[RelationSchema] = Field(..., description="List of the relations between two entities")
    # relation_scratchpad_thoughts_relations: str = Field(..., description="A scratch pad markdown table to list relations.")
    # relation_scratchpad_thoughts_critique: str = Field(..., description="A scratch pad markdown table to list critiques of the relation table.")
    relation_status: str = Field(..., description="Status of whether you were successful in extracting relations")


def extract_relations(doc):

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(

        openai_api_key=openai_api_key,

        temperature=0,

        # model="gpt-4-turbo",
        model="gpt-4o",
        # model="gpt-3.5-turbo-16k"
    )

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at extracting relations of named entities in documents to help form a Knowledge Graph.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    chain = prompt | llm.with_structured_output(
        schema=RelationListSchema,
        method="function_calling",
        include_raw=False,
    )

    content = ""

    paragraph_num = 0

    for paragraph in doc["paragraphs"]:

        paragraph_num += 1

        paragraph_text = paragraph["text"]

        content += "----------\n"
        content += f"Paragraph ID: {paragraph_num}\n"
        content += f"{paragraph_text}\n"

    entity_types = """
            (1) Person
    """

    relation_types = """
        (1) Friendship: between one Person and another Person
        (2) Friendly Behavior: from one Person directed to another Person
        (3) Enemy: between one Person and another Person
        (4) Evil Behavior: from one Person directed to another Person
    """

    relation_prompt = f"""
        A relation is a kind of relationship between entities.
        You are an expert in extracting relations from text.
       
        You will focus on relations of entity types from the list:
        {entity_types}
        
        An entity is always a single entity, not a group.
        Some relations tend to be symmetric, like: friendship.
        Some relations are not symmetric, like: employment.
        You must not include any incomplete relations in the results.
        You include both explicitly stated relations and relations that are implied but not explicitly stated.
        You must extract all of the unique relations you can find.
        The more the better!
        
        First, use the scratchpad to list your "thoughts" as you extract relations from the text.
        You list the relations you find and the paragraphs where you find them.
        
        Then, you summarize your thoughts into a unique set of relations you put into a "relation table". 
        Use markdown with columns:
            1) row index (1,2,3,...)
            2) relation type
            3) source entity
            4) target entity
                
        These rows are different and unique:
        | <index>         | Relation 1    | Person 1         | Person 2      |
        | <index>         | Relation 1    | Person 1      | Person 3         |
        
        But these rows are the same and not unique:
        | <index>         | Relation 1    | Person 1         | Person 2      |
        | <index>         | Relation 1    | Person 1      | Person 2         |
        
        Do not include any non-unique relations!
        
        Then, put each row of the "relation table" into your final results.
        Each row of the "relation table" maps to exactly one final result.       
        
        Organize the final results into a relation list of:
            (1) An index of the relation (1,2,3,...)
            (2) A label for the type of relation it is
            (3) A very short description of the relation and why you believe it is correct
            (4) The name of the entity that is the source of the relation
            (5) The name of the entity that is the target of the relation
            (6) Whether the relation is symmetric or not
            (7) The list of most supporting paragraph identifiers for the relation
        

        According to the criteria above, read the text below and extract unique relations of types:
        {relation_types}
        Only extract relations of the types from this list.
        Always include a short status of your final results.
        Text:    
        {content}
        """

    relation_prompt_short = f"""
            You are an expert in extracting entities and relations from text.

            You will focus on relations of entity types from the list:
                {entity_types}
        
            An entity should always be a single entity, not a group.
            
            You will extract relations of the types:
                 {relation_types}
                 
            Only extract relations of these types from this list.
            
            You must not include any incomplete relations in the results.
            
            You include:
                (1) Explicitly stated relations.
                (2) Relations that are implied by context but not explicitly stated.
               
            Organize the relations into a relation list of:
                (1) An index of the relation (1,2,3,...)
                (2) A label for the type of relation it is
                (3) A very short description of the relation and why you believe it is correct
                (4) The name of the entity that is the source of the relation
                (5) The name of the entity that is the target of the relation
                (6) The list of most supporting paragraph identifiers for the relation

            According to the criteria above, read the text below and extract the relations.
            Always include a short status of your final results.
            Text:    
            {content}
            """


    """
    Use the scratchpad to create a relation markdown table using these columns:
            "Row Index | Relation Type | Source Entity | Target Entity | Paragraph ID List"
                
            The Paragraph ID List reference Paragraphs that support the relation.
                                                    
            Then, critique the relation table to improve it.
            
            Create a markdown table for your critiques using these columns:
            "Row Index | Type of Change | Description of Change | NUmber of Relations Added/Removed"
               
            Use the critique table to list changes such as:
                Confirm that each row has a list of Paragraph IDs that support it
                Confirm that there are not other entities in the text with relations that were missed
                Merge relations that are the same except for the Paragraph IDs by merging the Paragraph IDs
                Remove redundant relations
                Confirm that the relations listed were accurately extracted
                Or, otherwise improve the list of relations.
                                        
            Then, use your critiques to make a new and final result list.
            """

    print(relation_prompt_short)

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{relation_prompt}"
        }
    )

    relation_map_list = []

    relation_list = output.relation_list

    relation_status = output.relation_status

    # relation_scratchpad_thoughts_relations = output.relation_scratchpad_thoughts_relations

    # relation_scratchpad_thoughts_critique = output.relation_scratchpad_thoughts_critique

    # relation_scratchpad_thoughts_entities = output.relation_scratchpad_thoughts_entities

    # relation_scratchpad_table = output.relation_scratchpad_table

    print(f"Relation Status: {relation_status}")

    # print(f"Relation Scratch Pad Thoughts:\n{relation_scratchpad_thoughts}")

    # print(f"Relation Scratch Pad Thoughts Entities:\n{relation_scratchpad_thoughts_entities}")

    # print(f"Relation Scratch Pad Thoughts Relations:\n{relation_scratchpad_thoughts_relations}")

    # print(f"Relation Scratch Pad Thoughts Critique:\n{relation_scratchpad_thoughts_critique}")

    # print(f"Relation Scratch Pad Table:\n{relation_scratchpad_table}")

    for relation in relation_list:

        paragraph_list = relation.paragraph_list

        paragraph_list_str = ", ".join([str(p) for p in paragraph_list])

        relation_map = {

            "relation_id": relation.relation_id,

            "relation_label": relation.relation_label,

            "relation_summary": relation.relation_summary,

            # "symmetric": relation.symmetric,

            "entity_source": relation.entity_source,

            "entity_target": relation.entity_target,

            "paragraph_list": paragraph_list_str
        }

        relation_map_list.append(relation_map)

    return relation_map_list


def main():

    print('KGraphGen Resolve Named Entity Test')

    docs = read_json_lines("../test_data_internal/hp_docs.2.jsonnl")

    # doc = docs[10]

    global_relation_map_list = []

    doc_num = 0

    for doc in docs:

        doc_num += 1

        relation_map_list = extract_relations(doc)

        for relation_dict in relation_map_list:
            print(json.dumps(relation_dict, indent=4))
            relation_dict['doc_num'] = doc_num
            global_relation_map_list.append(relation_dict)

    # with open("../test_output/hp_summary.2.jsonnl", "w", encoding="utf-8") as f:
    #    for entity_dict in entity_map_list:
    #        f.write(json.dumps(entity_dict) + "\n")

    with open("../test_output/hp_relation.2.jsonnl", "w", encoding="utf-8") as f:
        # for relation_dict in relation_map_list:
        for relation_dict in global_relation_map_list:
            f.write(json.dumps(relation_dict) + "\n")


if __name__ == "__main__":
    main()
