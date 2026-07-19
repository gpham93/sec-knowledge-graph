import os
import sys
import argparse
from rdflib import Graph
from google import genai
from google.genai.errors import APIError

def count_entities(g):
    # Counts parent corporations vs subsidiaries
    sec_corp = "http://enterprise.org/ontology/sec#Corporation"
    sec_sub = "http://enterprise.org/ontology/sec#Subsidiary"
    
    corps = list(g.subjects(predicate=None, object=Graph().value(None, None, sec_corp)))
    subs = list(g.subjects(predicate=None, object=Graph().value(None, None, sec_sub)))
    return len(corps), len(subs)

def main():
    parser = argparse.ArgumentParser(description="SEC Corporate Subsidiary Graph RAG Query Tool")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The natural language question to ask about the SEC filing data"
    )
    args = parser.parse_args()

    graph_file = "data_graph.ttl"
    if not os.path.exists(graph_file):
        print(f"Error: {graph_file} not found. Please run the extraction script first: python scripts/extract_sec_data.py")
        sys.exit(1)

    print("Loading corporate subsidiary knowledge graph...")
    g = Graph()
    g.parse(graph_file, format="turtle")
    
    # Calculate some stats for user awareness
    num_triples = len(g)
    print(f"Loaded {num_triples} triples successfully.")

    # Read the raw turtle ontology to feed as context to the LLM
    with open(graph_file, "r") as f:
        graph_context = f.read()

    # Check for Gemini API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "="*80)
        print("WARNING: GEMINI_API_KEY environment variable is not configured.")
        print("Graph RAG cannot generate an LLM response without an API key.")
        print("Please set your key: export GEMINI_API_KEY='your-gemini-key'")
        print("="*80)
        
        # Proactively answer using a fallback local search
        print("\n[Local Fallback Search] Analyzing entities in the graph...")
        query_lower = args.query.lower()
        
        # Simple keywords
        found = False
        if "subsidiary" in query_lower or "subsidiaries" in query_lower:
            print("\nList of Parent Companies and select Subsidiaries:")
            # Find all parent corporations and their hasName
            q = """
            PREFIX sec: <http://enterprise.org/ontology/sec#>
            SELECT ?parentName ?subName ?juris WHERE {
                ?parent a sec:Corporation ;
                        sec:hasName ?parentName ;
                        sec:ownsSubsidiary ?sub .
                ?sub sec:hasName ?subName ;
                     sec:hasJurisdiction ?juris .
            }
            LIMIT 15
            """
            for row in g.query(q):
                print(f" - {row.parentName} owns: {row.subName} ({row.juris})")
            found = True
        
        if "cik" in query_lower or "sic" in query_lower or "address" in query_lower:
            print("\nParent Corporation Metadata:")
            q = """
            PREFIX sec: <http://enterprise.org/ontology/sec#>
            SELECT ?name ?cik ?sic ?sicDesc ?state ?addr WHERE {
                ?parent a sec:Corporation ;
                        sec:hasName ?name ;
                        sec:cik ?cik ;
                        sec:sic ?sic ;
                        sec:sicDescription ?sicDesc ;
                        sec:stateOfIncorporation ?state ;
                        sec:businessAddress ?addr .
            }
            """
            for row in g.query(q):
                print(f"Corporate Name: {row.name}")
                print(f"  CIK: {row.cik}")
                print(f"  SIC: {row.sic} ({row.sicDesc})")
                print(f"  State: {row.state}")
                print(f"  Address: {row.addr.replace('\n', ', ')}")
            found = True
            
        if not found:
            print("\nCould not resolve query locally. Try asking about 'subsidiaries' or 'cik' / 'sic' metadata.")
        sys.exit(0)

    # Call Gemini model grounded in the knowledge graph context
    print("\nInitializing Gemini client and generating response...")
    try:
        client = genai.Client()
        
        prompt = f"""
You are an expert corporate intelligence analyst and senior data engineer.
Below is the Turtle (.ttl) representation of an OWL-governed knowledge graph representing parent companies and corporate subsidiaries extracted from SEC EDGAR 10-K filings.

Corporate Subsidiary Knowledge Graph (.ttl):
\"\"\"
{graph_context}
\"\"\"

Please answer the following user query based ONLY on the facts, relationships, and attributes declared in the knowledge graph context above.

User Query: "{args.query}"

Instructions:
1. Ground your answer strictly in the provided RDF graph facts. Do not assume or guess anything not asserted in the triples.
2. Explicitly cite the URIs and semantic relationships (e.g. sec:ownsSubsidiary, sec:isOwnedBy, sec:hasJurisdiction, etc.) used to construct your answer.
3. If the graph does not contain the answer, state that clearly.
"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        print("\n" + "="*80)
        print("GRAPH RAG RESPONSE:")
        print("="*80)
        print(response.text)
        print("="*80)
        
    except APIError as api_err:
        print(f"\nGemini API Error: {api_err}")
    except Exception as e:
        print(f"\nError running Graph RAG: {e}")

if __name__ == "__main__":
    main()
