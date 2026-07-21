
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

def create_prompt(entry):
    src_uri, src_label, tgt_uri, tgt_label, score, src_turtle, tgt_turtle = entry
    prompt = f'''
your task is to determine whethere these two concepts are the same ("Yes") or not ("No") by looking at 
their context
concept 1: {src_uri}
concept 2: {tgt_uri}
here is context on how concept 1 is defined:
{src_turtle}
here is context on how concept 2 is defined:
{tgt_turtle}
your task is ontology alignment, determining if these two are the same or not.
if they refer to different things say "No", here is a tricky example:
concept 1: costumer: a person who buys stuff from us
concept 2: costumer: a person who creates a account here
so the answer is no since they are different by how they are defined (this was tricky since they had the same label (costumer))
context (ontology) here defines how the concepts. 
in some cases there is no evidence to reject and and the have the same label so we say "Yes"
important: dont output anything other that "Yes" or "No"
    '''
    return prompt
def to_xml(input_file, output_file):
    g = Graph()
    g.parse(input_file)
    g.serialize(destination=output_file, format="xml")


def get_label(concept):
    """Return label or last part of URI."""
    if isinstance(concept, dict):
        label = concept.get("label") or concept.get("name") or ""
        if label:
            return label
        iri = concept.get("iri", "")
        if "/" in iri:
            return iri.split("/")[-1].split("#")[-1]
        return iri
    return str(concept)


def build_property_index(object_properties, data_properties):
    """class_iri -> [property_dict, ...]"""
    index = {}
    for prop in object_properties + data_properties:
        mentioned = set()
        for iri in prop.get("domain", []):
            mentioned.add(iri)
        for iri in prop.get("range", []):
            mentioned.add(iri)
        for iri in mentioned:
            if iri not in index:
                index[iri] = []
            index[iri].append(prop)
    return index


def extract_mini_ontology_turtle(concept_iri, property_index, full_graph):
    """
    Given a concept IRI and a property index, pull triples from the full
    rdflib Graph and return a Turtle string.
    """
    sub = Graph()
    for prefix, ns in full_graph.namespaces():
        sub.bind(prefix, ns)

    focus_uris = set()
    peripheral_uris = set()

    # the concept itself
    if concept_iri:
        focus_uris.add(URIRef(concept_iri))

    # related properties
    related_props = property_index.get(concept_iri, [])
    for prop in related_props:
        prop_iri = prop.get("iri", "")
        if prop_iri:
            focus_uris.add(URIRef(prop_iri))
        for d in prop.get("domain", []):
            if d and d.startswith("http"):
                peripheral_uris.add(URIRef(d))
        for r in prop.get("range", []):
            if r and r.startswith("http"):
                peripheral_uris.add(URIRef(r))

    peripheral_uris -= focus_uris

    # focus: all triples where URI is subject
    for uri in focus_uris:
        for s, p, o in full_graph.triples((uri, None, None)):
            sub.add((s, p, o))

    # peripheral: just skeleton
    skeleton_preds = [RDF.type, RDFS.label, RDFS.comment, SKOS.prefLabel, SKOS.altLabel]
    for uri in peripheral_uris:
        for pred in skeleton_preds:
            for s, p, o in full_graph.triples((uri, pred, None)):
                sub.add((s, p, o))

    return sub.serialize(format="turtle")


def build_candidates_list(
    candidate_pairs,
    source_items,
    target_items,
    source_obj_props,
    source_data_props,
    target_obj_props,
    target_data_props,
    source_graph,
    target_graph,
):
    """
    Build the candidates list. Each entry is:
    [source_uri, source_label, target_uri, target_label, score, source_turtle, target_turtle]
    """
    src_index = build_property_index(source_obj_props, source_data_props)
    tgt_index = build_property_index(target_obj_props, target_data_props)

    candidates = []
    for i, j, score in candidate_pairs:
        src = source_items[i]
        tgt = target_items[j]

        src_uri = src.get("iri", "")
        tgt_uri = tgt.get("iri", "")
        src_label = get_label(src)
        tgt_label = get_label(tgt)

        src_turtle = extract_mini_ontology_turtle(src_uri, src_index, source_graph)
        tgt_turtle = extract_mini_ontology_turtle(tgt_uri, tgt_index, target_graph)

        candidates.append([
            src_uri,
            src_label,
            tgt_uri,
            tgt_label,
            score,
            src_turtle,
            tgt_turtle,
        ])

    return candidates