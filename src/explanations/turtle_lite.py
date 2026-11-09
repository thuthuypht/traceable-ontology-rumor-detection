"""Dependency-free *restricted* Turtle reader for the two frozen provenance snapshots.

Supports exactly the constructs used by these snapshots: @prefix, qnames/IRIs,
`a`, semicolon predicate lists, comma object lists, string and typed-decimal
literals, integer/float scientific numeric literals. Rejects unknown syntax.

This is NOT a generic Turtle/RDF or OWL parser. For full RDF syntax/SHACL
validation, use rdflib in a separate environment; the numerical provenance
consistency audit can run without it.
"""
from __future__ import annotations
import re, ast
from collections import defaultdict

RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

class Namespace(str):
    def __new__(cls, base):
        return str.__new__(cls, base)
    def __getattr__(self, name):
        if name.startswith('_'): raise AttributeError(name)
        return str(self) + name
    def __getitem__(self, key):
        return str(self) + str(key)

class RDF:
    type = RDF_TYPE

URIRef = str

def split_unquoted(text, separator):
    """Split only separators outside double-quoted literals and angle-bracket IRIs."""
    parts=[];start=0;i=0;quoted=False;in_iri=False;escaped=False
    while i<len(text):
        ch=text[i]
        if quoted:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"': quoted=False
        elif in_iri:
            if ch=='>': in_iri=False
        else:
            if ch=='"': quoted=True
            elif ch=='<': in_iri=True
            elif text.startswith(separator,i):
                parts.append(text[start:i]);i+=len(separator);start=i;continue
        i+=1
    if quoted or in_iri: raise ValueError('Unclosed literal or IRI in Turtle')
    parts.append(text[start:])
    return parts

class Graph:
    def __init__(self):
        self._spo = defaultdict(lambda: defaultdict(list))
        self._triple_count = 0
        self.validation_mode = 'offline_restricted_turtle'

    def parse(self, *, data, format='turtle'):
        if format != 'turtle': raise ValueError('Only Turtle snapshot format supported')
        prefixes = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}
        lines=[]
        for lineno, line in enumerate(data.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'): continue
            if line.startswith('@prefix '):
                m=re.fullmatch(r'@prefix\s+(\w+):\s+<([^<>]+)>\s*\.',line)
                if not m: raise ValueError(f'Unsupported Turtle prefix on line {lineno}: {line[:100]}')
                prefixes[m.group(1)] = m.group(2)
                continue
            lines.append((lineno,line))
        chunks=[];cur=[]
        for lineno,line in lines:
            cur.append(line)
            if line.endswith(' .'):
                chunks.append((' '.join(cur)[:-2].strip(),lineno))
                cur=[]
        if cur: raise ValueError('Unterminated Turtle statement')

        def resource(s):
            s=s.strip()
            if s=='a': return RDF_TYPE
            if s.startswith('<') and s.endswith('>'):
                uri=s[1:-1]
                if not re.match(r'^https?://',uri): raise ValueError(f'Invalid IRI: {uri!r}')
                return uri
            m=re.fullmatch(r'([A-Za-z][\w-]*):([A-Za-z_][\w.\-]*)',s)
            if m and m.group(1) in prefixes:
                return prefixes[m.group(1)] + m.group(2)
            raise ValueError(f'Unsupported Turtle term: {s!r}')

        def objvalue(s):
            s=s.strip()
            if s.startswith('"'):
                m=re.fullmatch(r'"((?:[^"\\]|\\.)*)"(?:\^\^(?:[\w-]+:[\w-]+|<[^>]+>))?',s)
                if not m: raise ValueError(f'Unsupported literal: {s[:100]}')
                return ast.literal_eval('\"' + m.group(1) + '\"')
            if re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?',s):
                return str(float(s)) if ('e' in s.lower()) else s
            if s in {'true','false'}: return s
            return resource(s)

        for stmt,lineno in chunks:
            m=re.match(r'^(<[^<>]+>|[A-Za-z][\w-]*:[A-Za-z_][\w.\-]*)\s+(.+)$', stmt)
            if not m: raise ValueError(f'Unsupported Turtle subject in line {lineno}: {stmt[:120]}')
            subj=resource(m.group(1))
            rest=m.group(2)
            for clause in split_unquoted(rest, ' ; '):
                clause=clause.strip()
                if not clause: continue
                p=re.match(r'^(a|[A-Za-z][\w-]*:[A-Za-z_][\w.\-]*)\s+(.+)$',clause)
                if not p: raise ValueError(f'Unsupported predicate clause near line {lineno}: {clause[:120]}')
                pred=resource(p.group(1))
                for objpart in split_unquoted(p.group(2), ', '):
                    ob=objvalue(objpart)
                    self._spo[subj][pred].append(ob)
                    self._triple_count+=1
        return self

    def subjects(self, predicate, obj):
        for subj, props in self._spo.items():
            if obj in props.get(predicate, []):
                yield subj

    def objects(self, subj, predicate):
        yield from self._spo.get(subj, {}).get(predicate, [])

    def value(self, subj, predicate):
        values=self._spo.get(subj, {}).get(predicate, [])
        return values[0] if values else None

    def __len__(self):
        return self._triple_count
