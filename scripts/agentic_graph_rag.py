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


class FinancialTermExpander:
    """
    FINANCIAL DOMAIN TERM EXPANSION
    Maps user colloquial terms to SEC-standard 10-K filing terminology.
    """
    TERM_MAP = {
        "supply chain": ["third-party vendor", "outsourced service provider", "operational reliance", "cloud infrastructure risk", "supply chain"],
        "lawsuits": ["legal proceedings", "item 3", "contingent liabilities", "litigation", "lawsuit"],
        "lawsuit": ["legal proceedings", "item 3", "contingent liabilities", "litigation"],
        "executives": ["executive officers", "board of directors", "senior leadership", "item 10", "executive"],
        "subsidiaries": ["exhibit 21", "subsidiaries of the registrant", "owned entities", "subsidiary"]
    }

    @classmethod
    def expand_terms(cls, query_text: str) -> List[str]:
        query_lower = query_text.lower()
        expanded_terms = set(re.findall(r'\w+', query_text))
        for user_term, sec_terms in cls.TERM_MAP.items():
            if user_term in query_lower:
                expanded_terms.update(sec_terms)
        return [t for t in expanded_terms if len(t) > 2]


class DualPassQueryPlanner:
    """
    DUAL-PASS HYBRID EXECUTION PLANNER
    Detects multi-part prompts containing both structural questions (e.g. subsidiaries)
    and narrative text questions (e.g. risks/lawsuits) and splits them into parallel sub-tasks:
    - Sub-task A -> Graph (SPARQL) structural traversal for entity relationships.
    - Sub-task B -> Vector / Lexical text search with Financial Domain Term Expansion.
    """
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def execute_dual_pass(self, user_query: str) -> Tuple[str, List[Tuple[Any, Any, Any]], List[str]]:
        """
        Executes parallel sub-tasks and merges outputs into a unified context.
        """
        expanded_terms = FinancialTermExpander.expand_terms(user_query)

        # Sub-task A: Graph SPARQL / Multi-Hop Traversal
        seed_nodes = self.retriever.find_entry_point_nodes(expanded_terms)
        graph_triples = self.retriever.traverse_subgraph(seed_nodes, max_hops=2) if seed_nodes else []

        # Sub-task B: Expanded Vector/Lexical Search
        vector_matches = self.retriever.text_lexical_search(" ".join(expanded_terms))

        merged_context = "=== DUAL-PASS HYBRID RETRIEVAL CONTEXT ===\n"
        merged_context += f"SUB-TASK A (Structural Graph Triples - Count: {len(graph_triples)}):\n"
        for s, p, o in graph_triples[:40]:
            merged_context += f"<{s}> <{p}> {o.n3()}\n"
        
        merged_context += f"\nSUB-TASK B (Narrative Vector & Term Expansion Matches - Count: {len(vector_matches)}):\n"
        for match in vector_matches[:25]:
            merged_context += f"{match}\n"

        return merged_context, graph_triples, vector_matches


class AgenticRouter:
    """
    PHASE 3: AGENTIC ROUTING CONTROLLER
    Analyzes the user's prompt against the schema to determine optimal retrieval routing:
    - DUAL_PASS_HYBRID: Split into parallel Sub-task A (Graph) and Sub-task B (Vector + Term Expansion).
    - GRAPH_METADATA: Structured SPARQL/traversal for exact corporate attributes & tree links.
    - VECTOR_TEXT: Free-text lexical matching with term expansion.
    """

    def __init__(self, schema_text: str):
        self.schema_text = schema_text

    def route_query(self, user_query: str) -> str:
        """Determines the appropriate retrieval strategy."""
        query_lower = user_query.lower()

        structural_keywords = [
            "subsidiary", "subsidiaries", "owns", "owned", "parent", 
            "jurisdiction", "cik", "sic", "incorporated", "executive", "executives"
        ]
        narrative_keywords = [
            "risk", "risks", "supply chain", "lawsuit", "lawsuits", "litigation", "summary", "summarize"
        ]
        
        has_structural = any(kw in query_lower for kw in structural_keywords)
        has_narrative = any(kw in query_lower for kw in narrative_keywords)

        if has_structural and has_narrative:
            return "DUAL_PASS_HYBRID"
        elif has_structural:
            return "GRAPH_METADATA"
        else:
            return "VECTOR_TEXT"


class AgenticGraphRAGEngine:
    """
    ORCHESTRATOR & FALLBACK CONTROLLER
    Combines Schema Extractor, Hybrid Retriever, Dual-Pass Planner, and Agentic Router.
    """

    def __init__(self, graph_path: str = "data_graph.ttl"):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Knowledge graph file {graph_path} not found.")

        self.graph = Graph()
        self.graph.parse(graph_path, format="turtle")
        
        self.schema_extractor = GraphSchemaExtractor(self.graph)
        self.schema_text = self.schema_extractor.extract_schema()

        self.retriever = HybridRetriever(self.graph)
        self.planner = DualPassQueryPlanner(self.retriever)
        self.router = AgenticRouter(self.schema_text)

    def query(self, user_query: str) -> str:
        """Executes the Agentic Graph RAG pipeline end-to-end."""

        # 1. Route query
        route = self.router.route_query(user_query)
        print(f"[Agentic Router] Selected Routing Strategy: {route}")

        retrieved_context = ""
        triples_found = []

        if route == "DUAL_PASS_HYBRID":
            retrieved_context, triples_found, _ = self.planner.execute_dual_pass(user_query)
        elif route == "GRAPH_METADATA":
            expanded_keywords = FinancialTermExpander.expand_terms(user_query)
            seed_nodes = self.retriever.find_entry_point_nodes(expanded_keywords)
            if seed_nodes:
                triples_found = self.retriever.traverse_subgraph(seed_nodes, max_hops=2)

            if not triples_found:
                print("[Agentic Fallback] Graph traversal returned 0 structural triples. Triggering fallback to Vector Search...")
                text_matches = self.retriever.text_lexical_search(user_query)
                retrieved_context = "RETRIEVED TEXT MATCHES (FALLBACK):\n" + "\n".join(text_matches)
            else:
                retrieved_context = "RETRIEVED GRAPH SUBGRAPH (MULTI-HOP TRAVERSAL):\n"
                for s, p, o in triples_found[:50]:
                    retrieved_context += f"<{s}> <{p}> {o.n3()}\n"
        else:
            expanded_keywords = FinancialTermExpander.expand_terms(user_query)
            text_matches = self.retriever.text_lexical_search(" ".join(expanded_keywords))
            retrieved_context = "RETRIEVED LEXICAL VECTOR CONTEXT WITH FINANCIAL TERM EXPANSION:\n" + "\n".join(text_matches)

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
2. If the user prompt asks for structural entities (e.g. subsidiaries, executives), list them clearly.
3. If the user prompt asks for narrative summaries (e.g. risks, supply chain, lawsuits), summarize the key points.
4. Explicitly cite entity names, URIs, and relationships (e.g. sec:ownsSubsidiary, sec:hasExecutive, sec:reportsRisk).
"""

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
                model='gemini-flash-latest',
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

