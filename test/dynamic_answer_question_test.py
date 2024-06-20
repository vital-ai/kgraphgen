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


def load_config():
    with open("../agent_config.yaml", "r") as config_stream:
        try:
            return yaml.safe_load(config_stream)
        except yaml.YAMLError as exc:
            print("failed to load config file")


config = load_config()
openai_api_key = config['api_keys']['openai']


class QuestionAnswerSchema(BaseModel):
    """Answer to Question."""
    answer: str = Field(..., description="Answer to Question")


def answer_question(question: str, message: str, result_list: list):

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=50,
        return_messages=True
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model="gpt-4-turbo-preview")

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are an A.I. assistant.
                You are an expert at answering questions given input retrieved from a Knowledge Graph.  
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )

    chain = prompt | llm.with_structured_output(
        schema=QuestionAnswerSchema,
        method="function_calling",
        include_raw=False,
    )

    markdown_table = "| Entity Name | Relationship Type | Entity Name |\n"
    markdown_table += "|-------------|--------------------|--------------------|\n"

    for relationship in result_list:
        markdown_table += f"| {relationship['source_entity_name']} | {relationship['relationship_type']} | {relationship['target_entity_name']} |\n"

    question_prompt = f"""
    Given the question of:
    {question},
    and the status of retrieving relevant information from the knowledge graph:
    {message}
    and a table of the relevant results:
    {markdown_table}
    provide an answer to the question, or if you cannot answer the question
    given the provided information, reply with:
    "I cannot answer that question"
    """

    output = chain.invoke(
        {
            "chat_history": memory.buffer_as_messages,
            "input": f"{question_prompt}"
        }
    )

    answer = output.answer

    answer_map = {'answer': answer}

    return answer_map


def main():

    print('KGraphGen Dynamic Answer Question Test')

    # load in graph via RDF built using relationship test step

    # questions ask a question about an entity and relationship
    # with another entity or entities

    # the relationship source data is built via dynamic relationship
    # and the question step uses this as the input to answer the question

    # question criteria to include some of:
    # source entity type
    # source entity
    # relationship type
    # target entity type
    # target entity

    # graph is queried to find frames of interest
    # frames are provided as context to LLM to answer the question

    # example:
    # Who are Harry Potter's friends?
    # source entity: Harry Potter
    # source entity type: Person
    # relationship: friend
    # target entity type: Person (Person or Animal potentially)

    question = "Who are Harry Potter's friends?"

    message = "Relevant Results were found in the Knowledge Graph"

    result_list = [
        {
            'source_entity_name': 'Harry Potter',
            'relationship_type': 'friend-of',
            'target_entity_name': 'Ron Weasley'
        },
        {
            'source_entity_name': 'Harry Potter',
            'relationship_type': 'enemy-of',
            'target_entity_name': 'Lord Voldemort'
        },
        {
            'source_entity_name': 'Harry Potter',
            'relationship_type': 'friend-of',
            'target_entity_name': 'Hermione Granger'
        },
        {
            'source_entity_name': 'Harry Potter',
            'relationship_type': 'student-of',
            'target_entity_name': 'Professor Quirrell'
        },
        {
            'source_entity_name': 'Harry Potter',
            'relationship_type': 'classmate-of',
            'target_entity_name': 'Draco Malfoy'
        }
    ]

    answer_map = answer_question(question, message, result_list)

    print(answer_map)


if __name__ == "__main__":
    main()
