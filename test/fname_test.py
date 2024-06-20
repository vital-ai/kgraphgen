import json
import urllib

import nltk
from nltk.corpus import framenet as fn
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet as wn

nltk.download('framenet_v17')
nltk.download('verbnet3')
nltk.download('verbnet')
nltk.download('wordnet')



with open('../semlink/instances/vn-fn2.json') as f:
    vn_fn_mappings = json.load(f)


def map_framenet_to_verbnet(frame_name):
    verbnet_classes = []
    for vn_class, fn_frames in vn_fn_mappings.items():
        if frame_name in fn_frames:
            verbnet_classes.append(vn_class)
    return verbnet_classes


def map_framenet_lexical_unit_to_verbnet(lexical_unit):
    try:
        lus = fn.lus(name=lexical_unit)

        if not lus:
            print(f"No lexical unit found for: {lexical_unit}")
            return None

        luinfo = lus[0]

        print(f"Lexical unit info for {lexical_unit}: {luinfo}")

        frame = luinfo['frame']

        frame_name = frame.name

        print(f"Frame name for {lexical_unit}: {frame_name}")

        verbnet_classes = map_framenet_to_verbnet(frame_name)

        return verbnet_classes

    except KeyError as e:
        print(f"KeyError: {e} for lexical unit: {lexical_unit}")
        return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_synset_from_sense_key(sense_key):
    try:
        lemma = wn.lemma_from_key(sense_key)
        synset = lemma.synset()
        return synset
    except Exception as e:
        print(f"Error: Could not decode sense key '{sense_key}': {e}")
        return None


def main():

    print('FrameNet Test')

    # offset_id = "02427103"  # establish.v.01

    offset_id = "01647229"  # establish.v.02

    pos = 'v'

    synset = wn.synset_from_pos_and_offset(pos, int(offset_id))

    print(synset)

    sense_keys = ["found%2:41:00", "found%2:36:00"]

    synsets = {}
    for sense_key in sense_keys:
        synset = get_synset_from_sense_key(sense_key)
        if synset:
            synsets[sense_key] = synset

    # Print the results
    for key, synset in synsets.items():
        print(f"Sense key: {key}")
        print(f"WordNet synset: {synset}")
        print(f"Synset ID: {synset.offset()}")
        print(f"Definition: {synset.definition()}")
        print(f"Examples: {synset.examples()}")
        print()


    lexical_unit = "found.v"

    vn_classes = map_framenet_lexical_unit_to_verbnet(lexical_unit)

    if vn_classes:
        print(f"VerbNet classes for {lexical_unit}: {vn_classes}")
    else:
        print(f"No VerbNet classes found for {lexical_unit}.")


if __name__ == "__main__":
    main()
