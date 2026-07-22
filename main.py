
import re,ast, os
from ontoaligner.ontology import AgenticXDDataset
from ontoaligner.aligner.pruner.CandidateExtraction import prune_candidates
from rdflib import Graph
from helper import *
import os


#########################################################################
# Parameters
##########################################################################

threshold=0.95 # for pruner
LLM_choice = GPT5# or chat_with_Maverick -> LLM

#########################################################################
# Loading the dataset
##########################################################################

source_ontology_path="assets/AXD/ce-ce/BiOnto.rdf"
target_ontology_path="assets/AXD/ce-ce/CEON.rdf"
save_matches_in="assets/AXD/ce-ce/" #folder to save the results

task = AgenticXDDataset()
print("Creating the dataset ..")
dataset = task.collect(
    source_ontology_path=source_ontology_path,
    target_ontology_path=target_ontology_path,
)

#########################################################################
# Pruning the search space from n x m to k candidate pairs search with string similarity
##########################################################################
# separate source and target by type
Source_classes = [r for r in dataset['source'] if r['type'] == 'class']
Source_obj_props = [r for r in dataset['source'] if r['type'] == 'object property']
Source_data_props = [r for r in dataset['source'] if r['type'] == 'data property']

Target_classes = [r for r in dataset['target'] if r['type'] == 'class']
Target_obj_props = [r for r in dataset['target'] if r['type'] == 'object property']
Target_data_props = [r for r in dataset['target'] if r['type'] == 'data property']


class_pairs = prune_candidates(Source_classes, Target_classes, threshold=threshold)
obj_prop_pairs = prune_candidates(Source_obj_props, Target_obj_props, threshold=threshold)
data_prop_pairs = prune_candidates(Source_data_props, Target_data_props, threshold=threshold)

Candidates = extract_candidates(source_ontology_path=source_ontology_path,target_ontology_path=target_ontology_path,
        class_pairs=class_pairs, Source_classes=Source_classes, Target_classes=Target_classes,
        Source_obj_props=Source_obj_props, Source_data_props=Source_data_props,
        Target_obj_props=Target_obj_props, Target_data_props=Target_data_props,
        obj_prop_pairs=obj_prop_pairs,data_prop_pairs=data_prop_pairs,threshold=threshold)


#########################################################################
# Constructing the prompts for k candidates and asking the LLM
##########################################################################

LLM_decisions = []
for entry in Candidates:
    src_uri, src_label, tgt_uri, tgt_label, score, src_turtle, tgt_turtle = entry
    prompt = create_prompt(entry)
    decision = LLM_choice(prompt)
    yes_no = extract_yes_no_robust(decision)
    if yes_no not in ['yes','no']:
        yes_no = 'no'
    LLM_decisions.append((src_uri,tgt_uri,yes_no))

    print('='*70)
    print(src_uri)
    print(tgt_uri)
    print(yes_no)

save_alignment_xml(
    llm_decisions=LLM_decisions,
    output_file=save_matches_in+"alignment.xml",
    onto1_uri=source_ontology_path,
    onto2_uri=target_ontology_path,
    alignment_type="AXDAligner" 
)
