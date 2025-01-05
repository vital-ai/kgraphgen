from nltk.corpus import framenet as fn


def main():

    print('Hello World')

    count = 1
    for frame in fn.frames():
        print(f"({count}) Frame ID: {frame.ID}")
        print(f"Name: {frame.name}")
        print(f"Definition: {frame.definition}")

        frame_sentence_count = 0

        lu_list = [lu['name'] for lu in frame.lexUnit.values()]

        print(f"    Lexical Units: {lu_list}")

        # for lu in frame.lexUnit.values():
        #    for subCorpus in lu.subCorpus:
        #        sentence_count = len(subCorpus.sentence)
        #        frame_sentence_count += sentence_count

        # print(f"    Frame Sentence Count: {frame_sentence_count}")

        print("=====================================================")
        count += 1

    # exit(0)

    # Access a specific frame by its name
    frame_name = 'Commerce_buy'
    #     frame_name = 'Earnings_and_losses'
    frame = fn.frame(frame_name)
    print(f"Frame Name: {frame.name}")
    print(f"Definition: {frame.definition}")

    # Access frame elements
    print("Frame Elements:")
    for fe in frame.FE.values():
        # print(f"FE: {fe}")
        print(f"  {fe.name}: {fe.definition}")
        # print(f"{fe.semType}")

    # Access lexical units
    print("Lexical Units:")
    for lu in frame.lexUnit.values():
        print(f"  {lu.name}: {lu.definition}")

    # Access frame relations
    print("Frame Relations:")
    for relation in frame.frameRelations:
        print(f"  {relation.type.name}: {relation.superFrame.name} -> {relation.subFrame.name}")

    # Access annotated corpora
    # print("Annotated Corpora:")
    for lu in frame.lexUnit.values():
        for subCorpus in lu.subCorpus:
            # print(f"  Subcorpus: {subCorpus.name}")
            for sentence in subCorpus.sentence:
                # print(f"    Sentence: {sentence.text}")
                #print(f"    Sentence: {sentence}")

                for ann_set in sentence.annotationSet:
                    pass # print(ann_set)


if __name__ == "__main__":
    main()

