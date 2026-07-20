"""
Agentic Graph RAG System for SEC EDGAR Corporate Subsidiary Knowledge Graph
-----------------------------------------------------------------------------
Architecture Overview:
1. Phase 1 (Schema-First Injection): GraphSchemaExtractor extracts classes, 
   datatype properties, and object properties directly from the RDF triplestore.
2. Phase 2 (Hybrid Graph + Lexical Retrieval): HybridRetriever performs entry-point
   lexical/text search followed by parameterized 1-hop and 2-hop graph traversals.
3. Phase 3 (Agentic Routing & Fallbacks): AgenticRouter dynamically determines query 
   routing strategy (GRAPH_METADATA, VECTOR_TEXT, or HYBRID_PARALLEL) and 
   AgenticGraphRAGEngine executes zero-result fallback handling.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Any, Tuple
from rdflib import Graph, URIRef, Literal, RDF
from google import genai
from google.genai.errors import APIError


class GraphSchemaExtractor:
    """
    PHASE 1: SCHEMA-FIRST INJECTION
    Extracts the full cohesive schema model (Entity Classes, Datatype Properties, 
    and Object Properties/Relationships) from an rdflib.Graph triplestore.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def extract_schema(self) -> str:
        """Dynamically inspects the triplestore to extract schema axioms."""
        classes = set()
        object_properties = set()
        datatype_properties = set()

        # Query triples for schema definitions or inferred predicates
        for s, p, o in self.graph:
            if p == RDF.type:
                classes.add(str(o))
            if isinstance(o, URIRef) and p != RDF.type:
                object_properties.add(str(p))
            elif isinstance(o, Literal):
                datatype_properties.add(str(p))

        # Format clean schema representation for prompt injection
        schema_text = "=== KNOWLEDGE GRAPH FORMAL SCHEMA ===\n"
        schema_text += "Entity Classes:\n"
        for c in sorted(classes):
            schema_text += f"  - <{c}>\n"

        schema_text += "\nObject Properties (Structural Relationships):\n"
        for op in sorted(object_properties):
            schema_text += f"  - <{op}>\n"

        schema_text += "\nDatatype Properties (Entity Attributes):\n"
        for dp in sorted(datatype_properties):
            schema_text += f"  - <{dp}>\n"

        schema_text += "\nDomain Axioms:\n"
        schema_text += "  - sec:Corporation represents top-level parent companies (e.g. Goldman Sachs, Morgan Stanley, JPMorgan Chase).\n"
        schema_text += "  - sec:Subsidiary represents corporate subsidiaries owned by parent corporations.\n"
        schema_text += "  - sec:ownsSubsidiary is the inverse of sec:isOwnedBy.\n"
        schema_text += "  - Attribute predicates include: sec:cik, sec:sic, sec:sicDescription, sec:stateOfIncorporation, sec:businessAddress, sec:hasJurisdiction, sec:hasName.\n"

        return schema_text


class HybridRetriever:
    """
    PHASE 2: HYBRID GRAPH + VECTOR/LEXICAL RETRIEVAL
    Executes a 2-step hybrid search:
    1. Lexical/Vector match to find entry-point seed node URIs.
    2. Parameterized multi-hop graph traversal to expand context structurally from seed nodes.
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def find_entry_point_nodes(self, query_terms: List[str]) -> List[URIRef]:
        """Finds entry-point node URIs using fuzzy/lexical property matching."""
        seed_nodes = set()
        for term in query_terms:
            clean_term = term.strip().lower()
            if not clean_term or len(clean_term) < 2:
                continue

            for s, p, o in self.graph:
                if isinstance(o, Literal) and clean_term in str(o).lower():
                    seed_nodes.add(s)
                elif clean_term in str(s).lower():
                    seed_nodes.add(s)

        return list(seed_nodes)

    def traverse_subgraph(self, seed_nodes: List[URIRef], max_hops: int = 2) -> List[Tuple[Any, Any, Any]]:
        """Performs multi-hop graph expansion outward from seed node URIs."""
        visited_nodes = set(seed_nodes)
        subgraph_triples = set()

        current_frontier = set(seed_nodes)

        for hop in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                # Forward edges (Out-going)
                for p, o in self.graph.predicate_objects(subject=node):
                    subgraph_triples.add((node, p, o))
                    if isinstance(o, URIRef) and o not in visited_nodes:
                        visited_nodes.add(o)
                        next_frontier.add(o)

                # Reverse edges (In-coming)
                for s, p in self.graph.subject_predicates(object=node):
                    subgraph_triples.add((s, p, node))
                    if isinstance(s, URIRef) and s not in visited_nodes:
                        visited_nodes.add(s)
                        next_frontier.add(s)

            current_frontier = next_frontier

        return list(subgraph_triples)

    def text_lexical_search(self, query_text: str) -> List[str]:
        """Lexical free-text search across all graph literal assertions."""
        matches = []
        words = [w.lower() for w in re.findall(r'\w+', query_text) if len(w) > 2]
        
        for s, p, o in self.graph:
            if isinstance(o, Literal):
                val = str(o)
                val_lower = val.lower()
                if any(w in val_lower for w in words):
                    matches.append(f"<{s}> <{p}> \"{val}\"")
                    
        return matches[:30]


class AgenticRouter:
    """
    PHASE 3: AGENTIC ROUTING CONTROLLER
    Analyzes the user's prompt against the schema to determine optimal retrieval routing:
    - GRAPH_METADATA: Structured SPARQL/traversal for exact corporate attributes & tree links.
    - VECTOR_TEXT: Free-text lexical matching for unstructured queries.
    - HYBRID_PARALLEL: Blended traversal + text matching.
    """

    def __init__(self, schema_text: str):
        self.schema_text = schema_text

    def route_query(self, user_query: str) -> str:
        """Determines the appropriate retrieval strategy."""
        query_lower = user_query.lower()

        structural_keywords = [
            "subsidiary", "subsidiaries", "owns", "owned", "parent", 
            "jurisdiction", "cik", "sic", "incorporated", "address", "state"
        ]
        
        has_structural = any(kw in query_lower for kw in structural_keywords)

        if has_structural:
            if any(term in query_lower for term in ["list", "all", "where", "how many"]):
                return "HYBRID_PARALLEL"
            return "GRAPH_METADATA"
        else:
            return "VECTOR_TEXT"


class AgenticGraphRAGEngine:
    """
    ORCHESTRATOR & FALLBACK CONTROLLER
    Combines Schema Extractor, Hybrid Retriever, and Agentic Router with 
    automatic fallback logic if graph queries return zero results.
    """

    def __init__(self, graph_path: str = "data_graph.ttl"):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Knowledge graph file {graph_path} not found.")

        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        
        # Phase 1: Extract Schema
        self.schema_extractor = GraphSchemaExtractor(self.graph)
        self.schema_text = self.schema_extractor.extract_schema()

        # Phase 2: Hybrid Retriever
        self.retriever = HybridRetriever(self.graph)

        # Phase 3: Agentic Router
        self.router = AgenticRouter(self.schema_text)

    def query(self, user_query: str) -> str:
        """Executes the Agentic Graph RAG pipeline end-to-end."""

        # 1. Route query
        route = self.router.route_query(user_query)
        print(f"[Agentic Router] Selected Routing Strategy: {route}")

        retrieved_context = ""
        triples_found = []

        # Extract search keywords from prompt
        keywords = [w for w in re.findall(r'\w+', user_query) if len(w) > 2]

        if route in ["GRAPH_METADATA", "HYBRID_PARALLEL"]:
            # Perform multi-hop graph traversal
            seed_nodes = self.retriever.find_entry_point_nodes(keywords)
            if seed_nodes:
                triples_found = self.retriever.traverse_subgraph(seed_nodes, max_hops=2)

            # Check zero-result fallback logic
            if not triples_found:
                print("[Agentic Fallback] Graph traversal returned 0 structural triples. Triggering fallback to Vector/Text Search...")
                text_matches = self.retriever.text_lexical_search(user_query)
                retrieved_context = "RETRIEVED TEXT MATCHES (FALLBACK):\n" + "\n".join(text_matches)
            else:
                retrieved_context = "RETRIEVED GRAPH SUBGRAPH (MULTI-HOP TRAVERSAL):\n"
                for s, p, o in triples_found[:50]:
                    retrieved_context += f"<{s}> <{p}> {o.n3()}\n"

        if route == "VECTOR_TEXT" or not retrieved_context:
            text_matches = self.retriever.text_lexical_search(user_query)
            retrieved_context = "RETRIEVED LEXICAL TEXT CONTEXT:\n" + "\n".join(text_matches)

        # Build grounded prompt with Schema-First Injection
        prompt = f"""
You are an expert corporate intelligence analyst and senior knowledge engineer.

{self.schema_text}

RETRIEVED GROUNDING CONTEXT:
\"\"\"
{retrieved_context}
\"\"\"

User Query: "{user_query}"

Instructions:
1. Ground your answer strictly in the facts from the retrieved grounding context and schema above.
2. Explicitly cite entity names, URIs, and relationships (e.g. sec:ownsSubsidiary, sec:hasJurisdiction, sec:cik).
3. If the context does not contain the answer, state that clearly.
"""

        # Call Gemini API if available, else return structured text summary
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return (
                "\n" + "="*80 + "\n"
                "[AGENTIC GRAPH RAG OUTPUT - LOCAL FALLBACK MODE (GEMINI_API_KEY NOT SET)]\n"
                f"Query Route Selected: {route}\n"
                f"Retrieved Triples Count: {len(triples_found)}\n"
                "="*80 + "\n"
                "SCHEMA & CONTEXT PREVIEW:\n" + retrieved_context[:1000] + "\n" + "="*80
            )

        try:
            client = genai.Client()
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except APIError as e:
            return f"Gemini API Error: {e}"
        except Exception as e:
            return f"Error generating answer: {e}"

def main():
    parser = argparse.ArgumentParser(description="Agentic Graph RAG System CLI")
    parser.add_argument("--query", type=str, required=True, help="User prompt to process")
    args = parser.parse_args()

    engine = AgenticGraphRAGEngine("data_graph.ttl")
    result = engine.query(args.query)
    print("\n" + result)

if __name__ == "__main__":
    main()
