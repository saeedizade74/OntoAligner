
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
import re
from dotenv import load_dotenv
from openai import OpenAI
import re,ast, os
import openai
from openai import AzureOpenAI
import xml.etree.ElementTree as ET
from xml.dom import minidom

def save_alignment_xml(
        llm_decisions,
        output_file,
        onto1_uri,
        onto2_uri,
        relation="=",
        measure=1.0,
        alignment_type="??"
):

    # Register namespaces
    ALIGN_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
    RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    XSD_NS = "http://www.w3.org/2001/XMLSchema#"

    ET.register_namespace("", ALIGN_NS)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("xsd", XSD_NS)

    # Root RDF element
    rdf = ET.Element(
        f"{{{RDF_NS}}}RDF"
    )

    # Alignment element
    alignment = ET.SubElement(
        rdf,
        f"{{{ALIGN_NS}}}Alignment"
    )

    # Metadata
    ET.SubElement(alignment, "xml").text = "yes"
    ET.SubElement(alignment, "level").text = "0"
    ET.SubElement(alignment, "type").text = alignment_type

    ET.SubElement(alignment, "onto1").text = onto1_uri
    ET.SubElement(alignment, "onto2").text = onto2_uri

    ET.SubElement(alignment, "uri1").text = onto1_uri
    ET.SubElement(alignment, "uri2").text = onto2_uri


    # Add accepted mappings only
    for src_uri, tgt_uri, decision in llm_decisions:

        if str(decision).lower() != "yes":
            continue

        map_element = ET.SubElement(
            alignment,
            "map"
        )

        cell = ET.SubElement(
            map_element,
            "Cell"
        )

        ET.SubElement(
            cell,
            "entity1",
            {
                f"{{{RDF_NS}}}resource": src_uri
            }
        )

        ET.SubElement(
            cell,
            "entity2",
            {
                f"{{{RDF_NS}}}resource": tgt_uri
            }
        )

        measure_element = ET.SubElement(
            cell,
            "measure",
            {
                f"{{{RDF_NS}}}datatype": "xsd:float"
            }
        )

        measure_element.text = str(measure)

        ET.SubElement(
            cell,
            "relation"
        ).text = relation


    # Convert to pretty XML
    rough_xml = ET.tostring(
        rdf,
        encoding="utf-8"
    )

    pretty_xml = minidom.parseString(
        rough_xml
    ).toprettyxml(
        indent="\t",
        encoding="utf-8"
    )


    # Remove empty XML lines
    pretty_xml = b"\n".join(
        line for line in pretty_xml.splitlines()
        if line.strip()
    )


    with open(
        output_file,
        "wb"
    ) as f:
        f.write(pretty_xml)


    print(f"Alignment saved to: {output_file}")

load_dotenv()
def chat_with_Maverick(prompt):
  endpoint = os.getenv("endpoint_llama"); model_name = "Llama-4-Maverick-17B-128E-Instruct-FP8"
  deployment_name = "Llama-4-Maverick-17B-128E-Instruct-FP8-2"
  api_key = os.getenv("api_key_llama");client = OpenAI(base_url=f"{endpoint}",api_key=api_key)
  completion = client.chat.completions.create(model=deployment_name,
  messages=[{"role": "user", "content": prompt,}],);return completion.choices[0].message.content


def GPT5(prompt):
    client = AzureOpenAI(azure_endpoint = os.getenv("GPT5_endpoint"), api_version="2025-03-01-preview",api_key=os.getenv("GPT5_api_key"))
    response = client.chat.completions.create(model=  "GPT-5",messages = [{"role":"system",
    "content":prompt}],reasoning_effort ='high',stop=None)
    return response.choices[0].message.content # Access the content attribute

def extract_candidates(source_ontology_path,target_ontology_path,
        class_pairs, Source_classes, Target_classes,
        Source_obj_props, Source_data_props,
        Target_obj_props, Target_data_props,
        obj_prop_pairs,data_prop_pairs,threshold=0.75):
    # build the unified candidates list
    print("Loading RDF graphs ...")
    source_graph = Graph()
    source_graph.parse(source_ontology_path)
    target_graph = Graph()
    target_graph.parse(target_ontology_path)

    Candidates = []

    # class candidates
    Candidates += build_candidates_list(
        class_pairs, Source_classes, Target_classes,
        Source_obj_props, Source_data_props,
        Target_obj_props, Target_data_props,
        source_graph, target_graph,
    )

    # object property candidates
    Candidates += build_candidates_list(
        obj_prop_pairs, Source_obj_props, Target_obj_props,
        Source_obj_props, Source_data_props,
        Target_obj_props, Target_data_props,
        source_graph, target_graph,
    )

    # data property candidates
    Candidates += build_candidates_list(
        data_prop_pairs, Source_data_props, Target_data_props,
        Source_obj_props, Source_data_props,
        Target_obj_props, Target_data_props,
        source_graph, target_graph,
    )
    print(f"\nTotal candidates: {len(Candidates)}")
    print(f"  Class pairs:         {len(class_pairs)}")
    print(f"  Object prop pairs:   {len(obj_prop_pairs)}")
    print(f"  Data prop pairs:     {len(data_prop_pairs)}")


    return Candidates
def extract_yes_no_robust(text: str) -> str | None:
    """Extract a final Yes/No answer from LLM output."""

    text = text.strip()

    patterns = [
        # Highest confidence: explicit final answer markers
        r'(?:final answer|answer|output|decision|verdict|result|conclusion)\s*[:\-]?\s*(?:is\s*)?(?:\*\*|__|["\']|\$\\boxed\{)?\s*(yes|no)\s*(?:\}?\$|["\']|\*\*|__)?',

        # "The answer is No"
        r'the answer is\s*[:\-]?\s*(?:\*\*|__|["\']|\$\\boxed\{)?\s*(yes|no)',

        # "Therefore: Yes"
        r'(?:therefore|thus|hence)\s*,?\s*(?:the answer is\s*)?(yes|no)',

        # Markdown bold
        r'\*\*(yes|no)\*\*',

        # Markdown italic
        r'\*(yes|no)\*',

        # LaTeX boxed
        r'\\boxed\{\s*(yes|no)\s*\}',

        # Quoted
        r'["\'](yes|no)["\']',
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    # Fallback: inspect last few non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in reversed(lines[-5:]):
        # Remove common formatting
        cleaned = re.sub(
            r'[\*\_`\.\!\:\-\>\$\{\}\[\]\(\)"\']',
            '',
            line,
            flags=re.IGNORECASE,
        ).strip()

        m = re.fullmatch(r'(yes|no)', cleaned, re.IGNORECASE)
        if m:
            return m.group(1).lower()

    return None

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