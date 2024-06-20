

def main():

    print('KGraphGen Dynamic Question Generation Test')

    # initially this concerns only direct relations from entity A --> B
    # source entity and target entity may be arbitrary as
    # some relationships may be considered symmetrical like
    # friend-of
    # but other relationships are directed like:
    # person owns object

    # given input question concerning entities relating to other entities
    # determine criteria to include some of:
    # source entity type
    # source entity
    # relationship type
    # target entity type
    # target entity

    # covers cases like:
    # what did person123 buy?
    # who employed person123?
    # who are the friends of person123?
    # what animals does person123 have as pets?

    question = "Who are Harry Potter's friends?"

    # criteria:
    source_entity = "Harry Potter"
    source_entity_type = "Person"
    relationship_type = "friend-of"
    target_entity_type = "Person"


if __name__ == "__main__":
    main()
