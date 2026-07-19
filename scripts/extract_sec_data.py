from edgar import Company, set_identity
from rdflib import Graph, URIRef, Literal, Namespace, RDF, OWL
import re

set_identity("your.email@example.com") 
g = Graph()
SEC = Namespace("http://enterprise.org/ontology/sec#")
g.bind("sec", SEC)
g.bind("owl", OWL)

g.add((SEC.ownsSubsidiary, RDF.type, OWL.ObjectProperty))
g.add((SEC.isOwnedBy, RDF.type, OWL.ObjectProperty))
g.add((SEC.ownsSubsidiary, OWL.inverseOf, SEC.isOwnedBy))
g.add((SEC.isOwnedBy, RDF.type, OWL.FunctionalProperty))

ticker = "GS"
company = Company(ticker)
latest_10k = company.get_filings(form="10-K")[0].obj() 

parent_uri = URIRef(SEC + ticker)
g.add((parent_uri, RDF.type, SEC.Corporation))
g.add((parent_uri, SEC.hasName, Literal(company.name)))

for sub in latest_10k.subsidiaries:
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', sub.name.replace(" ", "_"))
    sub_uri = URIRef(SEC + clean_name)
    g.add((sub_uri, RDF.type, SEC.Subsidiary))
    g.add((sub_uri, SEC.hasName, Literal(sub.name)))
    g.add((parent_uri, SEC.ownsSubsidiary, sub_uri))

print(g.serialize(format="turtle"))
