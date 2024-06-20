from collections import deque
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
from langchain_community.chat_models import ChatAnyscale
from langchain_core.messages import SystemMessage
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
import asyncio
import requests
from bs4 import BeautifulSoup
import wikipediaapi
from urllib.parse import urlparse, unquote
import spacy

nlp = spacy.load("en_core_web_sm")


def load_config():
    with open("../agent_config.yaml", "r") as config_stream:
        try:
            return yaml.safe_load(config_stream)
        except yaml.YAMLError as exc:
            print("failed to load config file")


config = load_config()
anyscale_api_key = config['api_keys']['anyscale']

# model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"

# model_id = "mlabonne/NeuralHermes-2.5-Mistral-7B"

# model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

# model_id = "codellama/CodeLlama-70b-Instruct-hf"

base_url = "https://api.endpoints.anyscale.com/v1"

relation_extract_prompt = """
You are an expert at extracting Entities and Relations.
If no entities or relations are found, provide a status message.
You must respond using this JSON schema to structure your response:
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Entity and Relation Schema",
    "description": "A schema used to extract entities and relations from text.",
    "type": "object",
    "properties": {
        "status": {
            "description": "The status of extracting entities and relations",
            "type": "string"
        },
        "entities": {
            "description": "A list of entities with labels and types.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "description": "The name of the entity.",
                        "type": "string"
                    },
                    "entity-type": {
                        "description": "The type of the entity.",
                        "oneOf": [
                            {
                                "const": "PERSON",
                                "description": "A human being."
                            },
                            {
                                "const": "EVENT",
                                "description": "An event that occurs at a date, time, and location"
                            },
                            {
                                "const": "LOCATION",
                                "description": "A geographic place"
                            },
                            {
                                "const": "ORGANIZATION",
                                "description": "An organized group of people with a particular purpose."
                            },
                            {
                                "const": "PRODUCT",
                                "description": "A product or service that is commercially provided."
                            },
                            {
                                "const": "PHYSICAL_THING",
                                "description": "A physical thing."
                            },
                            {
                                "const": "DATE_TIME",
                                "description": "A date or time."
                            },
                            {
                                "const": "NUMBER",
                                "description": "A numerical or currency value."
                            },
                            {
                                "const": "OTHER",
                                "description": "Any other type of entity not covered by the above categories."
                            }
                        ]
                    }
                },
                "required": ["label", "entity-type"]
            }
        },
        "relations": {
            "description": "A list of relations between entities.",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "relation-type": {
                        "description": "The type of the relation.",
                        "oneOf": [
                            {
                                "const": "Org Based In",
                                "description": "An organization is based in a specific location."
                            },
                            {
                                "const": "Work For",
                                "description": "A person works for an organization."
                            },
                            {
                                "const": "Located In",
                                "description": "An entity is located in a Location."
                            },
                            {
                                "const": "Live In",
                                "description": "A person lives in a Location."
                            },
                            {
                                "const": "Provides",
                                "description": "An organization or person provides a product or service."
                            },
                            {
                                "const": "Occurred On",
                                "description": "An event occurred at a date or time"
                            },
                            {
                                "const": "Occurred In",
                                "description": "An event occurred in a Location"
                            },
                            {
                                "const": "Has Cost",
                                "description": "An organization, product, or service has a monetary cost"
                            },
                            {
                                "const": "Has Compensation",
                                "description": "An organization or person is compensated for a product, service, or labor"
                            }
                        ]
                    },
                    "subject": {
                        "description": "The subject entity of the relation.",
                        "type": "string"
                    },
                    "subject-type": {
                        "description": "The type of the subject",
                        "oneOf": [
                            {
                                "const": "PERSON",
                                "description": "A human being."
                            },
                            {
                                "const": "EVENT",
                                "description": "An event that occurs at a date, time, and location"
                            },
                            {
                                "const": "LOCATION",
                                "description": "A geographic place"
                            },
                            {
                                "const": "ORGANIZATION",
                                "description": "An organized group of people with a particular purpose."
                            },
                            {
                                "const": "PRODUCT",
                                "description": "A product or service that is commercially provided."
                            },
                            {
                                "const": "PHYSICAL_THING",
                                "description": "A physical thing."
                            },
                            {
                                "const": "DATE_TIME",
                                "description": "A date or time."
                            },
                            {
                                "const": "NUMBER",
                                "description": "A numerical or currency value."
                            },
                            {
                                "const": "OTHER",
                                "description": "Any other type of entity not covered by the above categories."
                            }
                        ]
                    }
                    "object": {
                        "description": "The object entity of the relation.",
                        "type": "string"
                    },
                    "object-type": {
                        "description": "The type of the object",
                        "oneOf": [
                            {
                                "const": "PERSON",
                                "description": "A human being."
                            },
                            {
                                "const": "EVENT",
                                "description": "An event that occurs at a date, time, and location"
                            },
                            {
                                "const": "LOCATION",
                                "description": "A geographic place"
                            },
                            {
                                "const": "ORGANIZATION",
                                "description": "An organized group of people with a particular purpose."
                            },
                            {
                                "const": "PRODUCT",
                                "description": "A product or service that is commercially provided."
                            },
                            {
                                "const": "PHYSICAL_THING",
                                "description": "A physical thing."
                            },
                            {
                                "const": "DATE_TIME",
                                "description": "A date or time."
                            },
                            {
                                "const": "NUMBER",
                                "description": "A numerical or currency value."
                            },
                            {
                                "const": "OTHER",
                                "description": "Any other type of entity not covered by the above categories."
                            }
                        ]
                    }
                },
                "required": ["relation-type", "subject", "object"]
            }
        }
    },
    "required": ["status"]
}
"""

#  "required": ["status", "entities", "relations"]


def is_valid_sentence(text):
    doc = nlp(text)

    # Define criteria to consider the text as a sentence
    if len(doc) < 3:  # Too short to be a sentence
        return False

    # Check if the text has a root that is a verb (indicating a clause)
    has_root_verb = any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc)
    return has_root_verb


def split_into_sentences(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if is_valid_sentence(sent.text)]
    return sentences


def get_model():

    model = model = ChatAnyscale(
        anyscale_api_base=base_url,
        model_name=model_id,
        verbose=True,
        anyscale_api_key=anyscale_api_key)

    return model


async def worker(context, sentence, model_instance, relation_extract_prompt, semaphore):

    async with semaphore:

        response_map = {
            "context": context,
            "sentence": sentence
        }

        message = f"""Using CONTEXT to help:\n---------\n{context}---------\n
        Extract the entities and relations you find in 
        TEXT:\n---------\n{sentence}
        """

        print(f"Processing Sentence: {sentence}")

        messages = [
            SystemMessage(content=relation_extract_prompt),
            HumanMessage(content=message),
        ]
        response = await model_instance.ainvoke(messages)

        try:

            content = response.content
            print(f"Response: {content}")

        except:
            pass

        response_map["response"] = response

        return response_map

lifo_queue = deque(maxlen=2)
lifo_queue.append("")
lifo_queue.append("")


async def create_tasks(sentence_list, model_instances, relation_extract_prompt, semaphore):
    tasks = []
    for i, sentence in enumerate(sentence_list):
        context = "\n".join(lifo_queue)
        lifo_queue.append(sentence)
        tasks.append(worker(context, sentence, model_instances[i % len(model_instances)], relation_extract_prompt, semaphore))
    return tasks


async def relation_extract(document_url):

    relation_list = []

    document_text = extract_wikipedia_text_from_url(document_url)

    # print(document_text)

    # exit(0)

    sentences = split_into_sentences(document_text)

    sent_num = 0

    for sentence in sentences:
        sent_num += 1
        print(f"({sent_num}) {sentence}")

    print(f"Sentence Count: {sent_num}")

    # sentence_list = sentences[:100]
    sentence_list = sentences

    num_workers = 20

    semaphore = asyncio.Semaphore(num_workers)

    model_instances = [get_model() for _ in range(num_workers)]

    tasks = await create_tasks(sentence_list, model_instances, relation_extract_prompt, semaphore)

    responses = await asyncio.gather(*tasks)

    for response_map in responses:

        # print(response)

        response = response_map["response"]

        content_json = response.content.strip()

        # print(content_json)

        try:

            parsed_json = json.loads(content_json)

            pretty_json = json.dumps(parsed_json, indent=4)

            print(pretty_json)

            response_map["parse_map"] = parsed_json

            relation_list.append(response_map)

        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"Error with:\n{content_json}")

    return relation_list


def extract_text_from_url(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the web page
        text = soup.get_text(separator=' ', strip=True)

        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None


def extract_wikipedia_text_from_url(url, lang='en'):
    # Parse the URL to get the path component
    parsed_url = urlparse(url)

    # Extract the title from the URL path
    title = unquote(parsed_url.path.split('/')[-1])

    user_agent = "MyWikipediaBot/1.0 (https://example.com/bot; myemail@example.com)"

    # Create a Wikipedia API object for the specified language
    wiki_wiki = wikipediaapi.Wikipedia(user_agent=user_agent)

    # Get the page by title
    page = wiki_wiki.page(title)

    # Check if the page exists
    if not page.exists():
        print(f"Page '{title}' does not exist.")
        return None

    # Extract the main text
    main_text = page.text

    return main_text


async def process_sentence(context, sentence, model_instance, relation_extract_prompt):

    message = f"""CONTEXT: {context}
    TEXT: {sentence}
    """

    print(f"Processing Sentence: {sentence}")

    messages = [
        SystemMessage(content=relation_extract_prompt),
        HumanMessage(content=message),
    ]

    response = await model_instance.ainvoke(messages)

    try:

        content = response.content
        print(f"Response: {content}")

    except:
        pass

    return response


def test_one():

    entity_list = []

    relation_list = []

    context = """In December 2015, Sam Altman, Greg Brockman, Reid Hoffman, Jessica Livingston, Peter Thiel, Elon Musk, Amazon Web Services (AWS), Infosys, and YC Research announced the formation of OpenAI and pledged over $1 billion to the venture. 
       The actual collected total amount of contributions was only $130 million until 2019.
       According to an investigation led by TechCrunch, Musk was its largest donor while YC Research did not contribute anything at all.
       The organization stated it would "freely collaborate" with other institutions and researchers by making its patents and research open to the public."""

    sentence = """OpenAI is headquartered at the Pioneer Building in Mission District, San Francisco."""

    model_instance = get_model()

    responses = asyncio.run(process_sentence(context, sentence, model_instance, relation_extract_prompt))

    for item in responses:
        if "relations" in item:
            relation_list.extend(item["relations"])
        if "status" in item:
            print(f"Status: {item['status']}")

    for relation in relation_list:

        subject = relation.get("subject")
        relation_type = relation.get("relation-type")
        obj = relation.get("object")

        if subject is not None and relation_type is not None and obj is not None:
            print(f"{subject} -- {relation_type} --> {obj}")


def main():

    print('KGraphGen Relation Extract Test')

    # test_one()
    # exit(0)

    # have N entity types in graph w/ embedding
    # have N relation types in graph w/ embedding

    # given sentence, find N closest entity types
    # given sentence, find N closest relation types

    # use entities associated with selected relation types
    # but compare with entity type list

    # construct prompt based on entity and relation types
    # parse sentence w/ prompt
    # get JSON back of extracted entity and relation types

    # prompt to include descriptions of entity and relation types

    # potentially fine tune model w/ datasets like rebel

    # if no entity or relation types are close to sentence, skip it

    entity_list = []

    relation_list = []

    document_url = "https://en.wikipedia.org/wiki/OpenAI"

    response_map_list = asyncio.run(relation_extract(document_url))

    works_for = []

    for response_map in response_map_list:

        response = response_map["response"]
        sentence = response_map["sentence"]
        context = response_map["context"]
        parse_map = response_map["parse_map"]

        if "relations" in parse_map:
            for relation in parse_map["relations"]:
                relation["sentence"] = sentence
                relation["context"] = context
                relation_list.append(relation)
        if "status" in parse_map:
            print(f"Status: {parse_map['status']}")

    print('-----------------------------------')

    for relation in relation_list:

        subject = relation.get("subject")
        relation_type = relation.get("relation-type")
        obj = relation.get("object")
        sentence = relation.get("sentence")
        context = relation.get("context")

        if subject is not None and relation_type is not None and obj is not None:
            print('==============================')
            print(f"{subject} -- {relation_type} --> {obj}")
            print(f"Sentence: {sentence}")
            print(f"Context:\n{context}")
            print('==============================')

        if relation_type == "Work For":
            works_for.append(relation)

    print('-----------------------------------')

    unique_relations_set = set()
    unique_relations = []

    for relation in works_for:
        subject = relation.get("subject")
        relation_type = relation.get("relation-type")
        obj = relation.get("object")
        relation_tuple = (subject, relation_type, obj)

        if relation_tuple not in unique_relations_set:
            unique_relations_set.add(relation_tuple)
            unique_relations.append(relation)

    print('-----------------------------------')

    for relation in unique_relations:

        subject = relation.get("subject")
        subject_type = relation.get("subject-type")
        relation_type = relation.get("relation-type")
        obj = relation.get("object")
        obj_type = relation.get("object-type")

        sentence = relation.get("sentence")
        context = relation.get("context")

        print(f"{subject}({subject_type}) -- {relation_type} --> {obj}({obj_type})")
        # print(f"Sentence: {sentence}")
        # print(f"Context: {context}\n")

    print('-----------------------------------')


"""
You extract relationships from a SENTENCE into relationship type and slot parameters.
Only use slot values explicitly stated in the text.  Do not infer unspecified values.
If a value is present, fill the slot.
If a value is not present specify the slot symbol: NO_VALUE
You may add any notes in the Notes field.
Use this structured format:
Relationship Type: [relationship type]
Slot Company: [company or NO_VALUE]
Slot Reporting Date: [reporting date or NO_VALUE]
Slot Reporting Amount Type: [profit or loss or NO_VALUE]
Slot Reporting Amount: [currency or NO_VALUE]
Slot Next Reporting Period: [reporting date or NO_VALUE]
Slot Projection Type for Next Period: [profit or loss or NO_VALUE]
Slot Projection Amount: [currency or NO_VALUE]
Slot Reporting Person: [person name or NO_VALUE]
Notes: [textual notes]
SENTENCE: The net loss of Microsoft in the second quarter of 2022 was $1 billion dollars.
"""


if __name__ == "__main__":
    main()


