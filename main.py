
from ontoaligner.ontology import AgenticXDDataset
from ontoaligner.aligner.pruner.CandidateExtraction import prune_candidates
from rdflib import Graph
from helper import to_xml, get_label, build_candidates_list, create_prompt

#########################################################################
# Loading the dataset
##########################################################################

source_ontology_path="assets/AXD/DPP/Logs4batch_id 1.xml"
target_ontology_path="assets/AXD/DPP/Logs4batch_id 2.xml"

task = AgenticXDDataset()
print("Creating the dataset ..")
dataset = task.collect(
    source_ontology_path=source_ontology_path,
    target_ontology_path=target_ontology_path,
)

#########################################################################
# Pruning the search space from n x m to k candidate pairs
# search with string similarity
##########################################################################
# separate source and target by type
Source_classes = [r for r in dataset['source'] if r['type'] == 'class']
Source_obj_props = [r for r in dataset['source'] if r['type'] == 'object property']
Source_data_props = [r for r in dataset['source'] if r['type'] == 'data property']

Target_classes = [r for r in dataset['target'] if r['type'] == 'class']
Target_obj_props = [r for r in dataset['target'] if r['type'] == 'object property']
Target_data_props = [r for r in dataset['target'] if r['type'] == 'data property']

# load rdf graphs once
print("Loading RDF graphs ...")
source_graph = Graph()
source_graph.parse(source_ontology_path)
target_graph = Graph()
target_graph.parse(target_ontology_path)

# prune candidates for all three types
print("Pruning candidates ...")
threshold=0.7
class_pairs = prune_candidates(Source_classes, Target_classes, threshold=threshold)
obj_prop_pairs = prune_candidates(Source_obj_props, Target_obj_props, threshold=threshold)
data_prop_pairs = prune_candidates(Source_data_props, Target_data_props, threshold=threshold)

# build the unified candidates list

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
#########################################################################
# Statistics of the pruned space (k candidatese)
##########################################################################
print(f"\nTotal candidates: {len(Candidates)}")
print(f"  Class pairs:         {len(class_pairs)}")
print(f"  Object prop pairs:   {len(obj_prop_pairs)}")
print(f"  Data prop pairs:     {len(data_prop_pairs)}")


# print samples
print("\n" + "=" * 70)

#########################################################################
# Constructing the prompts for k candidates and asking the LLM
##########################################################################

for entry in Candidates:
    src_uri, src_label, tgt_uri, tgt_label, score, src_turtle, tgt_turtle = entry
    # print("=" * 70)
    # print(f"\nSource: {src_label}  ({src_uri})")
    # print(f"Target: {tgt_label}  ({tgt_uri})")
    # print(f"Score:  {score:.3f}")
    # print("=" * 70)
    print(create_prompt(entry))
    