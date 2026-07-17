"""
created by Javad Saeedizade July 9th 2026
Returns all elements in the ontology.
Extending generic dataset (GenericOntology) by adding object and data properties
"""

from rdflib.collection import Collection
from rdflib import URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from typing import Any, Dict, List, Set
from tqdm import tqdm

from .generic import GenericOntology, OMDataset, track


class AgenticXDOntology(GenericOntology):

    def __init__(self, language: str = "en"):
        super().__init__(language=language)

    def _expand_class_expression(self, node):

        results: List[str] = []

        if isinstance(node, URIRef):
            results.append(str(node))
            return results

        if not isinstance(node, BNode):
            return results

        # unionOf
        union_list = self.graph.value(node, OWL.unionOf)
        if union_list is not None:
            try:
                col = Collection(self.graph, union_list)
                for item in col:
                    results.extend(self._expand_class_expression(item))
            except Exception:
                pass
            return list(dict.fromkeys(results))  
            
        # intersectionOf
        intersection_list = self.graph.value(node, OWL.intersectionOf)
        if intersection_list is not None:
            try:
                col = Collection(self.graph, intersection_list)
                for item in col:
                    results.extend(self._expand_class_expression(item))
            except Exception:
                pass
            return list(dict.fromkeys(results))

        # complementOf
        complement = self.graph.value(node, OWL.complementOf)
        if complement is not None:
            results.extend(self._expand_class_expression(complement))
            return list(dict.fromkeys(results))

        # Restriction (try to recover filler)
        if (node, RDF.type, OWL.Restriction) in self.graph:
            on_class = self.graph.value(node, OWL.onClass)
            if on_class is not None:
                results.extend(self._expand_class_expression(on_class))
            some_values_from = self.graph.value(node, OWL.someValuesFrom)
            if some_values_from is not None:
                results.extend(self._expand_class_expression(some_values_from))
            all_values_from = self.graph.value(node, OWL.allValuesFrom)
            if all_values_from is not None:
                results.extend(self._expand_class_expression(all_values_from))
            return list(dict.fromkeys(results))

        return list(dict.fromkeys(results))

    def get_property_domains(self, prop: URIRef) -> List[str]:
        """
        Returns a list of domain URIs for a given property, expanding unions/intersections where possible.
        """
        domains: List[str] = []
        for d in self.graph.objects(prop, RDFS.domain):
            domains.extend(self._expand_class_expression(d))
        # Deduplicate while preserving order
        return list(dict.fromkeys(domains))

    def get_property_ranges(self, prop: URIRef) -> List[str]:
        """
        Returns a list of range URIs for a given property, expanding unions/intersections where possible.
        """
        ranges: List[str] = []
        for r in self.graph.objects(prop, RDFS.range):
            ranges.extend(self._expand_class_expression(r))
        # Deduplicate while preserving order
        return list(dict.fromkeys(ranges))

    def get_property_info(self, prop: URIRef, prop_type: str):
        """
        Collects information for an object or data property.
        """
        if isinstance(prop, BNode):
            return None

        label = self.get_label(prop)
        name = self.get_name(prop)
        iri = self.get_iri(prop)

        if not iri:
            return None

        info: Dict[str, Any] = {
            "type": prop_type,
            "name": name,
            "iri": iri,
            "label": label,
            "synonyms": self.get_synonyms(prop),
            "comment": self.get_comments(prop),
            "domain": self.get_property_domains(prop),
            "range": self.get_property_ranges(prop),
        }
        return info

    def get_class_info(self, owl_class: URIRef):
        """
        Collects all relevant information for a given ontology class, adding the 'type' field.
        """
        base = super().get_class_info(owl_class)
        if base is None:
            return None
        base["type"] = "class"
        return base

    def extract_data(self, graph: Any):
        """
        Extracts classes and properties (object/data) with type info and, for properties, domain/range lists.
        """
        self.graph = graph
        parsed: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        # Classes
        for owl_class in tqdm(self.graph.subjects(RDF.type, OWL.Class)):
            if isinstance(owl_class, BNode):
                continue
            class_info = self.get_class_info(owl_class)
            if class_info:
                iri = class_info.get("iri")
                if iri and iri not in seen:
                    parsed.append(class_info)
                    seen.add(iri)

        # Object properties
        for obj_prop in tqdm(self.graph.subjects(RDF.type, OWL.ObjectProperty)):
            if isinstance(obj_prop, BNode):
                continue
            prop_info = self.get_property_info(obj_prop, "object property")
            if prop_info:
                iri = prop_info.get("iri")
                if iri and iri not in seen:
                    parsed.append(prop_info)
                    seen.add(iri)

        # Data properties
        for data_prop in tqdm(self.graph.subjects(RDF.type, OWL.DatatypeProperty)):
            if isinstance(data_prop, BNode):
                continue
            prop_info = self.get_property_info(data_prop, "data property")
            if prop_info:
                iri = prop_info.get("iri")
                if iri and iri not in seen:
                    parsed.append(prop_info)
                    seen.add(iri)

        return parsed


class AgenticXDDataset(OMDataset):
    """
    A dataset class similar to GenericOMDataset, but returning classes and also object/data properties,
    including each item's type and (for properties) domain and range lists.
    """
    track = track
    ontology_name = "AgenticXD-Source-Target"
    source_ontology = AgenticXDOntology()
    target_ontology = AgenticXDOntology()
