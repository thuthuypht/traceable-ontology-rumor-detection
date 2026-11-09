"""Run an independent SHACL engine on a separately supplied RDF file.

Requires: pip install rdflib pyshacl
Note: Parsing RDF != SHACL conformance; findings must be reported explicitly.
"""
import argparse
p=argparse.ArgumentParser()
p.add_argument('--ttl',required=True,help='Path to full RDF TTL kept in private research archive')
p.add_argument('--shapes',default='results/ontology216_faithfulness/provenance/ontology216_shapes.ttl')
a=p.parse_args()
try:
    from rdflib import Graph
    from pyshacl import validate
except ImportError as e:
    raise SystemExit('Install rdflib and pyshacl separately; no SHACL test was run: '+str(e))
dg=Graph().parse(a.ttl,format='turtle')
sg=Graph().parse(a.shapes,format='turtle')
conforms,report_graph,report_text=validate(dg,shacl_graph=sg,inference='none',abort_on_first=False)
print(report_text)
if not conforms:raise SystemExit('SHACL NONCONFORMANCE')
