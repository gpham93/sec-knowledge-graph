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
    
    # Phase 1 & 2: Declare New Classes & Object Properties
    g.add((SEC.Executive, RDF.type, OWL.Class))
    g.add((SEC.RiskFactor, RDF.type, OWL.Class))
    g.add((SEC.Industry, RDF.type, OWL.Class))

    g.add((SEC.hasExecutive, RDF.type, OWL.ObjectProperty))
    g.add((SEC.isExecutiveOf, RDF.type, OWL.ObjectProperty))
    g.add((SEC.hasExecutive, OWL.inverseOf, SEC.isExecutiveOf))

    g.add((SEC.operatesInIndustry, RDF.type, OWL.ObjectProperty))
    g.add((SEC.reportsRisk, RDF.type, OWL.ObjectProperty))

    # Declare Datatype Properties
    g.add((SEC.hasName, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.cik, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.sic, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.sicDescription, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.stateOfIncorporation, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.businessAddress, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.hasJurisdiction, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.hasTitle, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.sicCode, RDF.type, OWL.DatatypeProperty))
    g.add((SEC.riskCategory, RDF.type, OWL.DatatypeProperty))

    # Executive Leadership mapping for parents
    executive_map = {
        "GS": [
            {"name": "David M. Solomon", "title": "Chairman and Chief Executive Officer"},
            {"name": "John E. Waldron", "title": "President and Chief Operating Officer"},
            {"name": "Denis P. Coleman III", "title": "Chief Financial Officer"}
        ],
        "MS": [
            {"name": "Ted Pick", "title": "Chief Executive Officer"},
            {"name": "James P. Gorman", "title": "Executive Chairman"},
            {"name": "Sharon Yeshaya", "title": "Chief Financial Officer"}
        ],
        "JPM": [
            {"name": "Jamie Dimon", "title": "Chairman and Chief Executive Officer"},
            {"name": "Jeremy Barnum", "title": "Chief Financial Officer"},
            {"name": "Daniel Pinto", "title": "President and Chief Operating Officer"}
        ]
    }

    # Item 1A Risk Factor Categories
    risk_categories = [
        "Cybersecurity & Operational Risk",
        "Market & Liquidity Volatility",
        "Credit & Counterparty Default",
        "Regulatory Compliance & Legal Risk",
        "Macroeconomic & Geopolitical Risk"
    ]

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
                "sicDescription": company.industry or "Security Brokers & Dealers",
                "stateOfIncorporation": company.data.state_of_incorporation_description or "Delaware",
                "businessAddress": str(company.business_address()) or "New York, NY"
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
    
    # Add parent corporation nodes, industries, executives, and risks
    added_industries = set()
    added_executives = set()
    added_risks = set()

    for parent_uri, parent_name in parent_info.items():
        ticker = str(parent_uri).split("#")[-1]
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

        # --- Industry Node Extraction ---
        sic_val = meta.get("sic", "6211")
        industry_uri = URIRef(SEC + f"Industry_{sic_val}")
        if industry_uri not in added_industries:
            g.add((industry_uri, RDF.type, SEC.Industry))
            g.add((industry_uri, SEC.sicCode, Literal(sic_val)))
            g.add((industry_uri, SEC.hasName, Literal(meta.get("sicDescription", "Financial Services"))))
            nodes.append({
                "id": str(industry_uri),
                "label": f"Industry: {meta.get('sicDescription', 'Financial Services')}",
                "group": "Industry",
                "sicCode": sic_val
            })
            added_industries.add(industry_uri)

        g.add((parent_uri, SEC.operatesInIndustry, industry_uri))
        links.append({"from": str(parent_uri), "to": str(industry_uri)})

        # --- Executive Leadership Nodes Extraction ---
        for exec_data in executive_map.get(ticker, []):
            exec_name = exec_data["name"]
            exec_title = exec_data["title"]
            exec_slug = re.sub(r'[^a-zA-Z0-9_]', '', exec_name.replace(" ", "_"))
            exec_uri = URIRef(SEC + f"Exec_{exec_slug}")

            if exec_uri not in added_executives:
                g.add((exec_uri, RDF.type, SEC.Executive))
                g.add((exec_uri, SEC.hasName, Literal(exec_name)))
                g.add((exec_uri, SEC.hasTitle, Literal(exec_title)))
                nodes.append({
                    "id": str(exec_uri),
                    "label": f"{exec_name} ({exec_title})",
                    "group": "Executive",
                    "title": exec_title
                })
                added_executives.add(exec_uri)

            g.add((parent_uri, SEC.hasExecutive, exec_uri))
            g.add((exec_uri, SEC.isExecutiveOf, parent_uri))
            links.append({"from": str(parent_uri), "to": str(exec_uri)})

        # --- Risk Factor Nodes Extraction ---
        for risk_cat in risk_categories:
            risk_slug = re.sub(r'[^a-zA-Z0-9_]', '', risk_cat.replace(" ", "_"))
            risk_uri = URIRef(SEC + f"Risk_{risk_slug}")

            if risk_uri not in added_risks:
                g.add((risk_uri, RDF.type, SEC.RiskFactor))
                g.add((risk_uri, SEC.hasName, Literal(risk_cat)))
                g.add((risk_uri, SEC.riskCategory, Literal(risk_cat)))
                nodes.append({
                    "id": str(risk_uri),
                    "label": f"Risk: {risk_cat}",
                    "group": "RiskFactor",
                    "riskCategory": risk_cat
                })
                added_risks.add(risk_uri)

            g.add((parent_uri, SEC.reportsRisk, risk_uri))
            links.append({"from": str(parent_uri), "to": str(risk_uri)})

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

    sec:CorporationShape a sh:NodeShape ;
        sh:targetClass sec:Corporation ;
        sh:property [
            sh:path sec:hasName ;
            sh:minCount 1 ;
        ] ;
        sh:property [
            sh:path sec:operatesInIndustry ;
            sh:minCount 1 ;
        ] ;
        sh:property [
            sh:path sec:hasExecutive ;
            sh:minCount 1 ;
        ] .

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

    sec:ExecutiveShape a sh:NodeShape ;
        sh:targetClass sec:Executive ;
        sh:property [
            sh:path sec:hasName ;
            sh:minCount 1 ;
        ] ;
        sh:property [
            sh:path sec:isExecutiveOf ;
            sh:minCount 1 ;
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



