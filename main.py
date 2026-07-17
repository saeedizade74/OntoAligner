from ontoaligner.ontology import AgenticXDDataset
from helper import to_xml
task = AgenticXDDataset()

to_xml('datasets/DPP/Logs4batch_id 1.ttl',"datasets/DPP/Logs4batch_id 1.xml")
to_xml('datasets/DPP/Logs4batch_id 2.ttl',"datasets/DPP/Logs4batch_id 2.xml")

dataset = task.collect(
    source_ontology_path="datasets/DPP/Logs4batch_id 1.xml",
    target_ontology_path="datasets/DPP/Logs4batch_id 2.xml",
)

# print(dataset)
for row in dataset['source']:
    if 'prop' in row['type']:
        for k,v in row.items():
            print(k,v)