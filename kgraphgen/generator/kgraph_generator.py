from ai_haley_kg_domain.model.KGDocument import KGDocument
from kgraphservice.collection.kgraph_collection import KGraphCollection

from kgraphgen.pipeline.kgraph_pipeline import KGraphPipeline


class KGraphGenerator:
    def __int__(self):
        pass

    def generate_graph(self,
                       pipeline: KGraphPipeline,
                       kgcollection: KGraphCollection,
                       document: KGDocument) -> KGraphCollection:

        generated_collection = KGraphCollection()

        # Edge_hasKGDocumentSegment

        # hasKGDocumentSegmentTypeURI

        # KGDocumentSegmentType_DOCUMENT
        # KGDocumentSegmentType_MAJOR_SECTION
        # KGDocumentSegmentType_SECTION
        # KGDocumentSegmentType_PARAGRAPH
        # KGDocumentSegmentType_SENTENCE

        # KGDocument -- Edge_hasKGExtraction --> KGEntityMention

        # Edge_hasDocumentKGFrame

        # slot value entity --> entity mention

        # document-wide entities
        # KGDocument -- Edge_hasKGExtraction --> KGEntity

        # mention to entity
        # Edge_hasKGNormalizedEntity

        return generated_collection

