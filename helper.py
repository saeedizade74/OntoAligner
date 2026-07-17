from rdflib import Graph
def to_xml(input_file, output_file):
    g = Graph()
    g.parse(input_file)
    g.serialize(destination=output_file, format="xml")