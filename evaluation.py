import xml.etree.ElementTree as ET


def extract_alignments(xml_file):
    """
    Extract ontology alignments from:
    1. Alignment API XML
    2. RDF alignment files
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

    alignments = set()


    # -------------------------------
    # Case 1: Alignment API format
    # -------------------------------
    for cell in root.iter():

        if cell.tag.endswith("Cell"):

            entity1 = None
            entity2 = None

            for child in cell:

                if child.tag.endswith("entity1"):
                    entity1 = child

                elif child.tag.endswith("entity2"):
                    entity2 = child


            if entity1 is not None and entity2 is not None:

                uri1 = entity1.attrib.get(
                    f"{{{RDF_NS}}}resource"
                )

                uri2 = entity2.attrib.get(
                    f"{{{RDF_NS}}}resource"
                )

                if uri1 and uri2:
                    alignments.add(
                        frozenset([uri1, uri2])
                    )


    # -------------------------------
    # Case 2: RDF alignment format
    # -------------------------------
    if len(alignments) == 0:

        for elem in root.iter():

            attrs = elem.attrib

            # RDF triples often store:
            # source URI -> target URI

            if len(attrs) > 0:

                values = list(attrs.values())

                for v1 in values:
                    for v2 in values:

                        if (
                            v1.startswith("http")
                            and v2.startswith("http")
                            and v1 != v2
                        ):
                            alignments.add(
                                frozenset([v1, v2])
                            )


    return alignments



def evaluate_alignment(gold_file, predicted_file):

    gold = extract_alignments(gold_file)
    predicted = extract_alignments(predicted_file)


    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)


    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )


    print("=" * 60)
    print("Gold mappings      :", len(gold))
    print("Predicted mappings :", len(predicted))
    print("True positives     :", tp)
    print("False positives    :", fp)
    print("False negatives    :", fn)
    print("=" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")

    return precision, recall, f1



if __name__ == "__main__":

    gold_file = "assets/AXD/ce-ce/CEON-BiOnto.rdf"
    predicted_file = "assets/AXD/ce-ce/alignment.xml"

    evaluate_alignment(
        gold_file,
        predicted_file
    )