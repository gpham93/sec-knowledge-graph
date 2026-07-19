import os
# Ensure rate limit is explicitly configured to 9 requests per second
os.environ["EDGAR_RATE_LIMIT_PER_SEC"] = "9"

from edgar import Company, set_identity
from rdflib import Graph, URIRef, Literal, Namespace, RDF, OWL
import re
import json
import httpx
import pandas as pd
from splink import Linker, SettingsCreator, DuckDBAPI
import splink.comparison_library as cl
from pyshacl import validate
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==========================================
# 1. RATE LIMIT COMPLIANCE & RETRY LOGIC
# ==========================================


# Retry fetch operations with exponential backoff on transient HTTP/Network errors
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.NetworkError, httpx.TimeoutException)),
    reraise=True
)
def fetch_sec_filings(ticker_symbol):
    company = Company(ticker_symbol)
    filings = company.get_filings(form="10-K")
    if not filings:
        raise ValueError(f"No 10-K filings found for ticker: {ticker_symbol}")
    latest_10k_obj = filings[0].obj()
    return company, latest_10k_obj

# ==========================================
# 2. PROBABILISTIC ENTITY RESOLUTION
# ==========================================
def deduplicate_subsidiary_names(raw_names):
    # Ensure raw_names contains unique strings
    raw_names = list(set(raw_names))
    if len(raw_names) <= 1:
        return {name: name for name in raw_names}

    # Normalize name strings for similarity matching
    def clean_name(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df = pd.DataFrame({
        "unique_id": range(len(raw_names)),
        "original_name": raw_names,
        "name": [clean_name(n) for n in raw_names]
    })

    db_api = DuckDBAPI()
    
    # Configure Splink Jaro-Winkler / Levenshtein string matching
    name_comp = cl.LevenshteinAtThresholds("name", [2, 4])
    name_comp.configure(
        m_probabilities=[0.95, 0.03, 0.01, 0.01],
        u_probabilities=[0.001, 0.005, 0.01, 0.984]
    )

    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[name_comp],
        blocking_rules_to_generate_predictions=[],
        probability_two_random_records_match=0.1
    )

    linker = Linker(df, settings, db_api)
    predictions = linker.inference.predict()
    clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(predictions, 0.5)
    cluster_df = clustered.as_pandas_dataframe()

    # Map each original name to its cluster's representative name (first name in the cluster)
    rep_map = {}
    for _, group in cluster_df.groupby("cluster_id"):
        rep_name = group["original_name"].iloc[0]
        for orig in group["original_name"]:
            rep_map[orig] = rep_name
            
    return rep_map

# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def main():
    set_identity("your.email@example.com") 
    g = Graph()
    SEC = Namespace("http://enterprise.org/ontology/sec#")
    g.bind("sec", SEC)
    g.bind("owl", OWL)

    # Establish structural ontology properties
    g.add((SEC.ownsSubsidiary, RDF.type, OWL.ObjectProperty))
    g.add((SEC.isOwnedBy, RDF.type, OWL.ObjectProperty))
    g.add((SEC.ownsSubsidiary, OWL.inverseOf, SEC.isOwnedBy))
    g.add((SEC.isOwnedBy, RDF.type, OWL.FunctionalProperty))
    
    # Declare Datatype Properties
    g.add((SEC.hasName, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.cik, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.sic, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.sicDescription, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.stateOfIncorporation, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.businessAddress, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.hasJurisdiction, RDF.type, OWL.DatatypeProperty))

    # Pull data for multiple top-tier financial parents
    tickers = ["GS", "MS", "JPM"]
    all_subsidiaries = []      # List of tuples (parent_uri, sub_name, jurisdiction)
    parent_info = {}           # parent_uri -> company_name
    parent_metadata = {}       # parent_uri -> dict of meta values

    for ticker in tickers:
        print(f"Fetching filings for {ticker}...")
        try:
            company, latest_10k = fetch_sec_filings(ticker)
            parent_uri = URIRef(SEC + ticker)
            parent_info[parent_uri] = company.name
            
            # Cache parent metadata
            parent_metadata[parent_uri] = {
                "cik": str(company.cik),
                "sic": str(company.sic),
                "sicDescription": company.industry or "Unknown",
                "stateOfIncorporation": company.data.state_of_incorporation_description or "Unknown",
                "businessAddress": str(company.business_address()) or "Unknown"
            }
            
            for sub in latest_10k.subsidiaries:
                all_subsidiaries.append((parent_uri, sub.name, getattr(sub, "jurisdiction", None)))
        except Exception as e:
            print(f"Error fetching filings for {ticker}: {e}")

    # Collect all raw subsidiary names for cross-entity deduplication
    raw_sub_names = [sub_name for _, sub_name, _ in all_subsidiaries]
    rep_map = deduplicate_subsidiary_names(raw_sub_names)

    # Setup JSON graph structure for visualization
    nodes = []
    links = []
    
    # Add parent corporation nodes
    for parent_uri, parent_name in parent_info.items():
        meta = parent_metadata.get(parent_uri, {})
        g.add((parent_uri, RDF.type, SEC.Corporation))
        g.add((parent_uri, SEC.hasName, Literal(parent_name)))
        g.add((parent_uri, SEC.cik, Literal(meta.get("cik", ""))))
        g.add((parent_uri, SEC.sic, Literal(meta.get("sic", ""))))
        g.add((parent_uri, SEC.sicDescription, Literal(meta.get("sicDescription", ""))))
        g.add((parent_uri, SEC.stateOfIncorporation, Literal(meta.get("stateOfIncorporation", ""))))
        g.add((parent_uri, SEC.businessAddress, Literal(meta.get("businessAddress", ""))))
        
        nodes.append({
            "id": str(parent_uri),
            "label": parent_name,
            "group": "Corporation",
            "cik": meta.get("cik", ""),
            "sic": meta.get("sic", ""),
            "sicDescription": meta.get("sicDescription", ""),
            "stateOfIncorporation": meta.get("stateOfIncorporation", ""),
            "businessAddress": meta.get("businessAddress", "")
        })

    added_subs = set()
    added_edges = set()

    for parent_uri, sub_name, sub_jurisdict in all_subsidiaries:
        # Resolve names to their cluster's representative name
        resolved_name = rep_map.get(sub_name, sub_name)
        
        # URI normalization based on resolved name
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '', resolved_name.replace(" ", "_"))
        sub_uri = URIRef(SEC + clean_name)
        
        # Determine jurisdiction
        resolved_juris = sub_jurisdict or "Unknown"
        
        # Add the subsidiary node to the RDF graph if not already added
        if sub_uri not in added_subs:
            g.add((sub_uri, RDF.type, SEC.Subsidiary))
            g.add((sub_uri, SEC.hasName, Literal(resolved_name)))
            g.add((sub_uri, SEC.hasJurisdiction, Literal(resolved_juris)))
            
            nodes.append({
                "id": str(sub_uri), 
                "label": resolved_name, 
                "group": "Subsidiary",
                "jurisdiction": resolved_juris
            })
            added_subs.add(sub_uri)
            
        # Add the parent-subsidiary relationships (bi-directional)
        g.add((parent_uri, SEC.ownsSubsidiary, sub_uri))
        g.add((sub_uri, SEC.isOwnedBy, parent_uri))
        
        edge_key = (str(parent_uri), str(sub_uri))
        if edge_key not in added_edges:
            links.append({"from": str(parent_uri), "to": str(sub_uri)})
            added_edges.add(edge_key)

    # ==========================================
    # 3. SEMANTIC SHACL VALIDATION
    # ==========================================
    shacl_ttl = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix sec: <http://enterprise.org/ontology/sec#> .

    sec:SubsidiaryShape a sh:NodeShape ;
        sh:targetClass sec:Subsidiary ;
        sh:property [
            sh:path sec:hasName ;
            sh:minCount 1 ;
            sh:maxCount 1 ;
        ] ;
        sh:property [
            sh:path sec:isOwnedBy ;
            sh:minCount 1 ;
        ] ;
        sh:property [
            sh:path sec:hasJurisdiction ;
            sh:minCount 1 ;
            sh:maxCount 1 ;
        ] .
    """
    
    shacl_graph = Graph()
    shacl_graph.parse(data=shacl_ttl, format="turtle")

    conforms, results_graph, results_text = validate(
        g,
        shacl_graph=shacl_graph,
        ont_graph=None,
        inference='rdfs',
        abort_on_first=False
    )

    if not conforms:
        print("CRITICAL: SHACL validation failed!")
        print(results_text)
        raise ValueError(f"Semantic validation failed. Compliance report:\n{results_text}")

    # Serialize output formats
    g.serialize(destination="data_graph.ttl", format="turtle")
    with open("data_graph.json", "w") as json_file:
        json.dump({"nodes": nodes, "edges": links}, json_file, indent=2)

    print("Pipeline executed successfully: data_graph.ttl and data_graph.json generated.")

if __name__ == "__main__":
    main()


