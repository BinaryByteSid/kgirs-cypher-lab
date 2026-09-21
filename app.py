"""
Virtual Laboratory: Experiment 10 - Query Knowledge Graphs with Pattern-Based Queries (Streamlit)
Knowledge Graphs and Information Retrieval Systems (KGIRS)

Partitioned into the 4 core sections of the base lab template:
  1. Theory: Knowledge graph concepts, query clauses, objectives, procedure, and terminology.
  2. Simulation: Sample graph explorer, guided Query Builder, highlighted results, and trial logger.
  3. Quiz: Self-grading conceptual assessment with instant feedback.
  4. Report Generation: Student info, recorded query trials, observations, and downloadable PDF report.

Graph experiment note: A small in-memory knowledge graph (GRAPH_DATA) is loaded into a networkx
MultiDiGraph. The Query Builder produces pattern-based queries and executes the logic directly
against the networkx graph. Students can also edit queries by hand; a small read-only query
interpreter (section 2b) runs the edited query.

Note: No custom CSS is used so that Streamlit native light and dark themes render seamlessly.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT
# ======================================================================================

EXPERIMENT_CONFIG = {
    "experiment_no": "Experiment 10",
    "course": "Knowledge Graphs and Information Retrieval Systems (KGIRS)",
    "title": "Query Knowledge Graphs with Pattern-Based Queries",
    "objectives": [
        "Understand graph database concepts: nodes, relationships, properties, and labels.",
        "Write and interpret pattern-based queries using MATCH, WHERE, RETURN, ORDER BY and LIMIT.",
        "Perform multi-hop traversals and filtered pattern matches over a knowledge graph.",
        "Retrieve and interpret meaningful information (lists, paths, counts) from a knowledge graph."
    ]
}

THEORY_CONTENT = {
    "background": """
### Overview & Principles
A **knowledge graph** stores facts as a network of connected entities. Instead of rows in tables,
information is modelled as **nodes** (the entities - people, movies, companies, cities) joined by
**relationships** (the facts that link them - *acted in*, *directed*, *located in*). This lab uses
the **property graph model**, a standard way to structure graph data:

- **Node**: an entity, e.g. `Keanu Reeves` or `The Matrix`.
- **Label**: a type tag that groups nodes, written with a colon, e.g. `:Person`, `:Movie`.
- **Relationship**: a *directed*, *typed* connection between two nodes, e.g. `ACTED_IN`.
  Every relationship has a start node, an end node and exactly one type.
- **Property**: a key-value pair stored on a node **or** a relationship,
  e.g. `born: 1964` on a person, or `role: "Neo"` on an `ACTED_IN` relationship.

### Pattern-Based Query Language
Graph queries use a **declarative, pattern-based approach**. *Declarative* means you describe
**what** pattern you are looking for, and the engine decides **how** to find it. Patterns are
drawn with ASCII-art:

```
(a:Person)-[:ACTED_IN]->(m:Movie)
```

- `( )` is a node, `a` is a variable, `:Person` is a label.
- `-[ ]->` is a relationship; `:ACTED_IN` is its type and `->` is its direction.
- `<-[ ]-` reverses the direction, and `-[ ]-` ignores direction.
- `-[*1..3]-` is a **variable-length** relationship: any path of 1 to 3 hops.
- `{name: "The Matrix"}` inside a node is an inline property filter.

### Core Query Clauses
| Clause | Purpose | Example |
|---|---|---|
| `MATCH` | Declares the graph pattern to find | `MATCH (p:Person)-[:DIRECTED]->(m:Movie)` |
| `WHERE` | Filters the matched patterns | `WHERE p.born > 1970` |
| `RETURN` | Chooses what to output (nodes, properties, aggregates) | `RETURN p.name, m.name` |
| `ORDER BY` | Sorts the result rows | `ORDER BY m.released DESC` |
| `LIMIT` | Caps the number of rows returned | `LIMIT 5` |

Aggregate functions such as `count()` group rows automatically by the other returned values, e.g.
`MATCH (n)-[r]->() RETURN n.name, count(r)` counts outgoing relationships per node.

### Workflow & System Overview
1. **Graph Construction**: The sample knowledge graph (16 nodes, 25 relationships) is loaded from
   plain Python data into an in-memory `networkx.MultiDiGraph`.
2. **Query Building**: A guided Query Builder turns your choices (pattern mode, labels,
   relationship types, filters, hop count) into pattern-based queries, which you can also
   edit by hand.
3. **Execution**: The same pattern is evaluated directly on the networkx graph (label checks,
   property comparisons, edge-direction checks, and depth-limited path search). Edited queries
   are run by a built-in query interpreter that supports read-only MATCH patterns.
4. **Result Analysis**: Results are shown as a table and as a highlighted subgraph, and each
   query can be logged as a trial for your report.
    """,
    "procedure": [
        "Step 1: Review the theory on nodes, relationships, properties, labels, and query clauses.",
        "Step 2: Open the Simulation section and explore the sample movie knowledge graph (graph view, node list, relationship list, and schema).",
        "Step 3: In the Query Builder, choose a query pattern mode (Node Lookup, 1-hop Traversal, Multi-hop Traversal, Filtered Pattern Match, or Aggregation) and set its options.",
        "Step 4: Read the generated query pattern and predict what it should return before looking at the results. Optionally edit the query and run your own version.",
        "Step 5: Examine the result table and the highlighted subgraph, and compare them with your prediction.",
        "Step 6: Click 'Record Current Trial' to log the query mode, query text, and result counts.",
        "Step 7: Repeat with at least one query from each pattern mode (vary labels, directions, hop counts, and filters).",
        "Step 8: Complete the Quiz, then open Report Generation, enter your details and observations, and download the PDF report."
    ],
    "key_terms": {
        "Node": "An entity in the graph (e.g. a person or a movie), drawn as ( ).",
        "Relationship / Edge": "A directed, typed connection between a start node and an end node, drawn as -[ ]->.",
        "Property": "A key-value pair stored on a node or relationship, e.g. born: 1964 or role: \"Neo\".",
        "Label": "A tag that classifies nodes into types, e.g. :Person, :Movie, :Organization, :City.",
        "Relationship Type": "The single name that describes what a relationship means, e.g. ACTED_IN, DIRECTED.",
        "Pattern-Based Query": "A declarative approach where you describe a pattern (shape) you're looking for, and the engine finds all matches.",
        "Pattern Matching": "Finding every part of the graph that has the same shape as the pattern in a MATCH clause.",
        "Traversal": "Moving from a node to its neighbours by following relationships.",
        "Multi-hop Query": "A query that follows two or more relationships in sequence, e.g. actor -> movie -> studio.",
        "Variable-length Path": "A relationship pattern with a hop range, e.g. -[*1..3]-, that matches paths of several lengths."
    }
}

# Sample knowledge graph: a mini movie domain (people, movies, studios, cities).
# Facts are simplified for teaching. Node "label" is the node type; every other key is a property.
GRAPH_DATA = {
    "domain": "Mini movie knowledge graph: people, movies, studios, and cities",
    "nodes": [
        {"id": "p1", "label": "Person", "name": "Keanu Reeves", "born": 1964},
        {"id": "p2", "label": "Person", "name": "Carrie-Anne Moss", "born": 1967},
        {"id": "p3", "label": "Person", "name": "Laurence Fishburne", "born": 1961},
        {"id": "p4", "label": "Person", "name": "Lana Wachowski", "born": 1965},
        {"id": "p5", "label": "Person", "name": "Christopher Nolan", "born": 1970},
        {"id": "p6", "label": "Person", "name": "Leonardo DiCaprio", "born": 1974},
        {"id": "p7", "label": "Person", "name": "Tom Hardy", "born": 1977},
        {"id": "p8", "label": "Person", "name": "Emma Thomas", "born": 1971},
        {"id": "m1", "label": "Movie", "name": "The Matrix", "released": 1999},
        {"id": "m2", "label": "Movie", "name": "John Wick", "released": 2014},
        {"id": "m3", "label": "Movie", "name": "Inception", "released": 2010},
        {"id": "m4", "label": "Movie", "name": "The Dark Knight Rises", "released": 2012},
        {"id": "o1", "label": "Organization", "name": "Warner Bros", "founded": 1923},
        {"id": "o2", "label": "Organization", "name": "Syncopy", "founded": 2001},
        {"id": "c1", "label": "City", "name": "Los Angeles", "country": "USA"},
        {"id": "c2", "label": "City", "name": "London", "country": "UK"},
    ],
    "relationships": [
        {"source": "p1", "target": "m1", "type": "ACTED_IN", "properties": {"role": "Neo"}},
        {"source": "p2", "target": "m1", "type": "ACTED_IN", "properties": {"role": "Trinity"}},
        {"source": "p3", "target": "m1", "type": "ACTED_IN", "properties": {"role": "Morpheus"}},
        {"source": "p1", "target": "m2", "type": "ACTED_IN", "properties": {"role": "John Wick"}},
        {"source": "p6", "target": "m3", "type": "ACTED_IN", "properties": {"role": "Cobb"}},
        {"source": "p7", "target": "m3", "type": "ACTED_IN", "properties": {"role": "Eames"}},
        {"source": "p7", "target": "m4", "type": "ACTED_IN", "properties": {"role": "Bane"}},
        {"source": "p4", "target": "m1", "type": "DIRECTED", "properties": {}},
        {"source": "p5", "target": "m3", "type": "DIRECTED", "properties": {}},
        {"source": "p5", "target": "m4", "type": "DIRECTED", "properties": {}},
        {"source": "p8", "target": "m3", "type": "PRODUCED", "properties": {}},
        {"source": "p8", "target": "m4", "type": "PRODUCED", "properties": {}},
        {"source": "o2", "target": "m3", "type": "PRODUCED", "properties": {}},
        {"source": "o2", "target": "m4", "type": "PRODUCED", "properties": {}},
        {"source": "o1", "target": "m1", "type": "DISTRIBUTED", "properties": {}},
        {"source": "o1", "target": "m3", "type": "DISTRIBUTED", "properties": {}},
        {"source": "o1", "target": "m4", "type": "DISTRIBUTED", "properties": {}},
        {"source": "p5", "target": "o2", "type": "FOUNDED", "properties": {"year": 2001}},
        {"source": "p8", "target": "o2", "type": "FOUNDED", "properties": {"year": 2001}},
        {"source": "o1", "target": "c1", "type": "LOCATED_IN", "properties": {}},
        {"source": "o2", "target": "c2", "type": "LOCATED_IN", "properties": {}},
        {"source": "p5", "target": "c2", "type": "BORN_IN", "properties": {}},
        {"source": "p7", "target": "c2", "type": "BORN_IN", "properties": {}},
        {"source": "p8", "target": "c2", "type": "BORN_IN", "properties": {}},
        {"source": "p6", "target": "c1", "type": "BORN_IN", "properties": {}},
    ]
}

SIMULATION_CONFIG = {
    "query_modes": {
        "Node Lookup": "Node Lookup",
        "Relationship Traversal (1-hop)": "1-Hop Traversal",
        "Multi-hop Traversal": "Multi-Hop",
        "Filtered Pattern Match": "Pattern Match",
        "Aggregation": "Aggregation",
    },
    "hop_options": [2, 3],
    "aggregation_modes": [
        "Outgoing relationships per node",
        "All relationships per node (degree)",
        "Relationships per type",
        "Nodes per label",
    ],
    # Node colors follow the label (fixed order, never re-assigned by rank); shape is a second cue.
    "label_styles": {
        "Person": {"color": "#2a78d6", "symbol": "circle"},
        "Movie": {"color": "#eb6834", "symbol": "square"},
        "Organization": {"color": "#1baf7a", "symbol": "diamond"},
        "City": {"color": "#4a3aa7", "symbol": "triangle-up"},
    },
    "highlight_color": "#e34948",
    "edge_color": "rgba(137, 135, 129, 0.65)",
    "faded_edge_color": "rgba(137, 135, 129, 0.18)",
    "bar_color": "#2a78d6",
    "faded_opacity": 0.2,
    "node_size": 22,
    "focus_size": 32,
    "arrow_gap": 0.05,
    "arrow_length": 0.03,
    "layout_seed": 7,
    # Hand-placed coordinates (x, y) keep the small graph readable; unlisted nodes fall back to spring layout.
    "node_positions": {
        "m2": (0.3, 7.0), "p1": (1.3, 5.6), "m1": (2.4, 4.0), "p2": (0.2, 3.6),
        "p3": (1.0, 1.8), "p4": (3.0, 1.8), "o1": (4.6, 4.0), "c1": (4.6, 1.0),
        "p6": (7.0, 1.0), "m3": (7.0, 3.2), "m4": (7.0, 5.6), "p7": (5.8, 7.2),
        "p5": (9.2, 2.2), "p8": (9.2, 6.6), "o2": (10.4, 4.4), "c2": (12.2, 4.4),
    },
    "graph_height": 560,
    "max_limit": 50,
}

ANY = "(any)"
NO_FILTER = "(no filter)"
NUMERIC_OPERATORS = ["=", "<>", ">", ">=", "<", "<="]
STRING_OPERATORS = ["=", "<>", "CONTAINS", "STARTS WITH", "ENDS WITH"]
DIRECTIONS = {
    "Outgoing  (a)-[r]->(b)": "out",
    "Incoming  (a)<-[r]-(b)": "in",
    "Either  (a)-[r]-(b)": "both",
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which statement best describes a knowledge graph?",
        "options": [
            "A) A spreadsheet in which every row is a document and every column is a keyword",
            "B) A network of entities (nodes) connected by typed relationships, with properties describing both",
            "C) A chart that plots numeric values against time",
            "D) A relational table with a single primary key and no foreign keys"
        ],
        "answer_index": 1,
        "explanation": "A knowledge graph models facts as entities joined by meaningful, typed relationships; in the property graph model both can carry key-value properties."
    },
    {
        "id": 2,
        "question": "In the pattern (k:Person)-[:ACTED_IN]->(m:Movie), what is ACTED_IN?",
        "options": [
            "A) A node label",
            "B) A property key",
            "C) A relationship type",
            "D) A variable name"
        ],
        "answer_index": 2,
        "explanation": "Inside square brackets, the name after the colon is the relationship type. Person and Movie are node labels; k and m are variables."
    },
    {
        "id": 3,
        "question": "In (k)-[:ACTED_IN {role: \"Neo\"}]->(m), where is the property 'role' stored?",
        "options": [
            "A) On the node k",
            "B) On the node m",
            "C) As an extra label on both nodes",
            "D) On the ACTED_IN relationship itself"
        ],
        "answer_index": 3,
        "explanation": "The braces sit inside the relationship brackets, so role is a relationship property: it describes this particular acting fact, not the actor or the movie alone."
    },
    {
        "id": 4,
        "question": "Which option correctly describes the roles of MATCH, WHERE and RETURN?",
        "options": [
            "A) MATCH declares the pattern, WHERE filters the matches, RETURN chooses what to output",
            "B) MATCH creates new nodes, WHERE deletes nodes, RETURN saves the graph",
            "C) MATCH sorts rows, WHERE limits rows, RETURN counts rows",
            "D) All three clauses are interchangeable and can be written in any order"
        ],
        "answer_index": 0,
        "explanation": "MATCH describes the shape to find, WHERE adds conditions on the matched variables, and RETURN projects the nodes, properties or aggregates you want back."
    },
    {
        "id": 5,
        "question": "Using this lab's sample graph, what does MATCH (p:Person) WHERE p.born > 1970 RETURN p.name return?",
        "options": [
            "A) The names of all eight people in the graph",
            "B) Leonardo DiCaprio, Tom Hardy and Emma Thomas",
            "C) Christopher Nolan, Leonardo DiCaprio, Tom Hardy and Emma Thomas",
            "D) A single number: the count of people born after 1970"
        ],
        "answer_index": 1,
        "explanation": "Only people with born strictly greater than 1970 match (1974, 1977, 1971). Christopher Nolan was born in 1970, which fails the > test. RETURN p.name lists names, not a count."
    },
    {
        "id": 6,
        "question": "In this lab's graph, what does MATCH (a:Person)<-[:DIRECTED]-(b) WHERE a.name = \"Christopher Nolan\" RETURN b.name return?",
        "options": [
            "A) Inception and The Dark Knight Rises",
            "B) Syncopy",
            "C) London",
            "D) No rows, because no DIRECTED relationship points into Christopher Nolan"
        ],
        "answer_index": 3,
        "explanation": "The arrow points into a, so the pattern needs someone who directed Nolan. His DIRECTED relationships go out to his movies, so the incoming pattern matches nothing. Direction matters."
    },
    {
        "id": 7,
        "question": "What does the pattern (a)-[r]-(b), written without an arrow head, match?",
        "options": [
            "A) Only relationships that go from a to b",
            "B) Only relationships that go from b to a",
            "C) Relationships between a and b in either direction",
            "D) Pairs of nodes that are not connected at all"
        ],
        "answer_index": 2,
        "explanation": "Leaving out the arrow makes the pattern direction-agnostic: a stored relationship in either direction matches."
    },
    {
        "id": 8,
        "question": "Which pattern matches paths of 1 to 3 relationships of any type between a and b?",
        "options": [
            "A) (a)-[:3]->(b)",
            "B) (a)-[*1..3]-(b)",
            "C) (a)-[1-3]-(b)",
            "D) (a)-{1,3}-(b)"
        ],
        "answer_index": 1,
        "explanation": "Variable-length relationships use an asterisk and a range: [*min..max]. With no type given, any relationship type may be used on each hop."
    },
    {
        "id": 9,
        "question": "In this lab's graph, starting from Keanu Reeves and ignoring direction, which node is reached in exactly 2 hops but not in 1?",
        "options": [
            "A) Warner Bros (Keanu Reeves - The Matrix - Warner Bros)",
            "B) The Matrix",
            "C) John Wick",
            "D) London"
        ],
        "answer_index": 0,
        "explanation": "The Matrix and John Wick are direct (1-hop) neighbours. Warner Bros distributed The Matrix, so it is 2 hops away. London is not reachable within 2 hops."
    },
    {
        "id": 10,
        "question": "What does MATCH (n)-[r]->() RETURN n.name, count(r) AS rels ORDER BY rels DESC LIMIT 3 return?",
        "options": [
            "A) The three relationship types that occur most often",
            "B) Every node, sorted alphabetically, with its property count",
            "C) The three nodes with the most outgoing relationships, highest count first",
            "D) Three random nodes from the graph"
        ],
        "answer_index": 2,
        "explanation": "count(r) groups rows by n.name and counts each node's outgoing relationships. ORDER BY rels DESC puts the largest counts first, and LIMIT 3 keeps only the top three rows."
    }
]


# ======================================================================================
# 2. SIMULATION ENGINE: IN-MEMORY KNOWLEDGE GRAPH & QUERY EXECUTION
# ======================================================================================

@st.cache_resource
def build_knowledge_graph() -> nx.MultiDiGraph:
    """Loads GRAPH_DATA into a networkx MultiDiGraph (built once and shared; never mutated)."""
    graph = nx.MultiDiGraph()
    for node in GRAPH_DATA["nodes"]:
        props = {k: v for k, v in node.items() if k not in ("id", "label")}
        graph.add_node(node["id"], label=node["label"], props=props)
    for i, rel in enumerate(GRAPH_DATA["relationships"], start=1):
        graph.add_edge(rel["source"], rel["target"], key=f"r{i}",
                       type=rel["type"], props=dict(rel.get("properties", {})))
    return graph


@st.cache_data
def compute_layout(seed: int, fixed_positions: tuple) -> dict:
    """Fixed 2-D positions for every node so the graph looks the same across reruns."""
    graph = build_knowledge_graph()
    fixed = dict(fixed_positions)
    if not fixed:
        pos = nx.spring_layout(nx.Graph(graph), seed=seed, k=0.9, iterations=300)
        return {n: (float(xy[0]), float(xy[1])) for n, xy in pos.items()}
    xs = [p[0] for p in fixed.values()]
    ys = [p[1] for p in fixed.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    initial = {n: ((x - cx) / scale, (y - cy) / scale) for n, (x, y) in fixed.items() if n in graph}
    pos = nx.spring_layout(nx.Graph(graph), seed=seed, pos=initial or None,
                           fixed=list(initial) or None, k=0.9, iterations=300)
    return {n: (float(xy[0]), float(xy[1])) for n, xy in pos.items()}


def node_name(graph, node_id) -> str:
    return graph.nodes[node_id]["props"]["name"]


def node_label(graph, node_id) -> str:
    return graph.nodes[node_id]["label"]


def node_display(graph, node_id) -> str:
    return f"{node_name(graph, node_id)} ({node_label(graph, node_id)})"


def sorted_node_ids(graph, label=ANY) -> list:
    order = list(SIMULATION_CONFIG["label_styles"])
    ids = [n for n in graph.nodes if label == ANY or node_label(graph, n) == label]
    return sorted(ids, key=lambda n: (order.index(node_label(graph, n)), node_name(graph, n)))


def graph_labels(graph) -> list:
    order = list(SIMULATION_CONFIG["label_styles"])
    return sorted({node_label(graph, n) for n in graph.nodes}, key=order.index)


def relationship_types(graph) -> list:
    return sorted({d["type"] for _, _, d in graph.edges(data=True)})


def property_keys(graph, label) -> list:
    keys = set()
    for n in sorted_node_ids(graph, label):
        keys.update(graph.nodes[n]["props"])
    return sorted(keys, key=lambda k: (k != "name", k))


def property_values(graph, label, key) -> list:
    values = {graph.nodes[n]["props"][key] for n in sorted_node_ids(graph, label)
              if key in graph.nodes[n]["props"]}
    return sorted(values)


def cypher_literal(value) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def node_pattern(var, label) -> str:
    return f"({var})" if label == ANY else f"({var}:{label})"


def format_props(props: dict) -> str:
    if not props:
        return ""
    return "{" + ", ".join(f"{k}: {cypher_literal(v)}" for k, v in props.items()) + "}"


def edge_text(graph, u, v, data) -> str:
    props = format_props(data["props"])
    props = f" {props}" if props else ""
    return f"({node_name(graph, u)})-[:{data['type']}{props}]->({node_name(graph, v)})"


def compare_values(actual, operator, expected) -> bool:
    """Pattern-based comparison; a missing property (null) never matches."""
    if actual is None:
        return False
    if operator in ("CONTAINS", "STARTS WITH", "ENDS WITH"):
        actual, expected = str(actual), str(expected)
        if operator == "CONTAINS":
            return expected in actual
        if operator == "STARTS WITH":
            return actual.startswith(expected)
        return actual.endswith(expected)
    if isinstance(actual, str) != isinstance(expected, str):
        return False
    return {
        "=": actual == expected,
        "<>": actual != expected,
        ">": actual > expected,
        ">=": actual >= expected,
        "<": actual < expected,
        "<=": actual <= expected,
    }[operator]


def with_limit(cypher: str, limit: int) -> str:
    return f"{cypher}\nLIMIT {limit}" if limit else cypher


def apply_limit(rows: list, limit: int) -> list:
    return rows[:limit] if limit else rows


def make_result(mode, scope, cypher, rows, columns, nodes, edges, summary, focus=None, chart=None) -> dict:
    return {
        "mode": mode,
        "scope": scope,
        "cypher": cypher,
        "table": pd.DataFrame(rows, columns=columns),
        "nodes": set(nodes),
        "edges": set(edges),
        "focus": set(focus or []),
        "summary": summary,
        "chart": chart,
    }


def query_node_lookup(graph, label, prop_key, operator, value, limit) -> dict:
    """MATCH (n:Label {key: value}) / MATCH (n:Label) WHERE n.key <op> value."""
    pattern = node_pattern("n", label)
    if prop_key == NO_FILTER:
        cypher = f"MATCH {pattern}\nRETURN n\nORDER BY n.name"
        condition = "no property filter"
    elif operator == "=":
        inline = f"{pattern[:-1]} {{{prop_key}: {cypher_literal(value)}}})"
        cypher = f"MATCH {inline}\nRETURN n\nORDER BY n.name"
        condition = f"{prop_key} = {cypher_literal(value)}"
    else:
        condition = f"{prop_key} {operator} {cypher_literal(value)}"
        cypher = f"MATCH {pattern}\nWHERE n.{condition}\nRETURN n\nORDER BY n.name"
    cypher = with_limit(cypher, limit)

    matched = []
    for n in sorted_node_ids(graph, label):
        props = graph.nodes[n]["props"]
        if prop_key == NO_FILTER or compare_values(props.get(prop_key), operator, value):
            matched.append(n)
    matched = apply_limit(sorted(matched, key=lambda n: node_name(graph, n)), limit)

    rows = []
    for n in matched:
        other = {k: v for k, v in graph.nodes[n]["props"].items() if k != "name"}
        rows.append([node_name(graph, n), node_label(graph, n), format_props(other)])

    target = "node(s)" if label == ANY else f":{label} node(s)"
    scope = ("All" if label == ANY else label) + ("" if prop_key == NO_FILTER else f" | {condition}")
    summary = f"Found {len(matched)} {target} with {condition}."
    return make_result("Node Lookup", scope, cypher, rows, ["Node", "Label", "Other Properties"],
                       matched, [], summary)


def query_one_hop(graph, start_id, rel_type, direction, limit) -> dict:
    """MATCH (a)-[r:TYPE]->(b) WHERE a.name = "..." (direction-aware, one hop)."""
    left, right = {"out": ("-", "->"), "in": ("<-", "-"), "both": ("-", "-")}[direction]
    rel = "[r]" if rel_type == ANY else f"[r:{rel_type}]"
    start_name = node_name(graph, start_id)
    cypher = (
        f"MATCH {node_pattern('a', node_label(graph, start_id))}{left}{rel}{right}(b)\n"
        f"WHERE a.name = {cypher_literal(start_name)}\n"
        "RETURN a.name AS start, type(r) AS relationship, b.name AS neighbour, labels(b) AS label\n"
        "ORDER BY neighbour, relationship"
    )
    cypher = with_limit(cypher, limit)

    hits = []
    if direction in ("out", "both"):
        for u, v, k, d in graph.out_edges(start_id, keys=True, data=True):
            hits.append((v, (u, v, k), d, "outgoing"))
    if direction in ("in", "both"):
        for u, v, k, d in graph.in_edges(start_id, keys=True, data=True):
            hits.append((u, (u, v, k), d, "incoming"))
    hits = [h for h in hits if rel_type == ANY or h[2]["type"] == rel_type]
    hits = apply_limit(sorted(hits, key=lambda h: (node_name(graph, h[0]), h[2]["type"])), limit)

    rows = [[start_name, h[2]["type"], h[3], node_name(graph, h[0]), node_label(graph, h[0]),
             edge_text(graph, h[1][0], h[1][1], h[2])] for h in hits]
    nodes = {start_id} | {h[0] for h in hits}
    edges = {h[1] for h in hits}
    rel_desc = "any type" if rel_type == ANY else rel_type
    dir_desc = {"out": "outgoing", "in": "incoming", "both": "incoming or outgoing"}[direction]
    summary = (f"{start_name} has {len(hits)} {dir_desc} "
               f"relationship(s) of {rel_desc}, reaching {len(nodes) - 1} distinct neighbour(s).")
    scope = f"{start_name} | {rel_desc} | {direction}"
    return make_result("1-Hop Traversal", scope, cypher, rows,
                       ["Start", "Relationship", "Direction", "Neighbour", "Neighbour Label", "Stored As"],
                       nodes, edges, summary, focus=[start_id])


def find_paths(graph, start_id, max_hops) -> list:
    """All undirected paths of 1..max_hops from start_id; no relationship repeats in a path."""
    paths = []

    def walk(node, node_seq, step_seq, used):
        if step_seq:
            paths.append((list(node_seq), list(step_seq)))
        if len(step_seq) == max_hops:
            return
        steps = [(v, (u, v, k), d, "->") for u, v, k, d in graph.out_edges(node, keys=True, data=True)]
        steps += [(u, (u, v, k), d, "<-") for u, v, k, d in graph.in_edges(node, keys=True, data=True)]
        for nxt, edge, data, arrow in steps:
            if edge in used:
                continue
            used.add(edge)
            node_seq.append(nxt)
            step_seq.append((edge, data["type"], arrow))
            walk(nxt, node_seq, step_seq, used)
            step_seq.pop()
            node_seq.pop()
            used.discard(edge)

    walk(start_id, [start_id], [], set())
    return paths


def path_text(graph, node_seq, step_seq) -> str:
    parts = [node_name(graph, node_seq[0])]
    for (_, rel_type, arrow), nxt in zip(step_seq, node_seq[1:]):
        link = f" -[:{rel_type}]-> " if arrow == "->" else f" <-[:{rel_type}]- "
        parts.append(link + node_name(graph, nxt))
    return "".join(parts)


def query_multi_hop(graph, start_id, max_hops, target_label, limit) -> dict:
    """MATCH path = (a)-[*1..k]-(b) WHERE a.name = "..." RETURN path."""
    start_name = node_name(graph, start_id)
    cypher = (
        f"MATCH path = {node_pattern('a', node_label(graph, start_id))}-[*1..{max_hops}]-{node_pattern('b', target_label)}\n"
        f"WHERE a.name = {cypher_literal(start_name)} AND b <> a\n"
        "RETURN length(path) AS hops, b.name AS endNode, path\n"
        "ORDER BY hops, endNode"
    )
    cypher = with_limit(cypher, limit)

    found = []
    for node_seq, step_seq in find_paths(graph, start_id, max_hops):
        end = node_seq[-1]
        if end == start_id or (target_label != ANY and node_label(graph, end) != target_label):
            continue
        found.append((len(step_seq), node_name(graph, end), path_text(graph, node_seq, step_seq),
                      node_seq, step_seq))
    found = apply_limit(sorted(found, key=lambda p: (p[0], p[1], p[2])), limit)

    rows = [[p[0], p[1], node_label(graph, p[3][-1]), p[2]] for p in found]
    nodes = {start_id} | {n for p in found for n in p[3]}
    edges = {step[0] for p in found for step in p[4]}
    ends = {p[3][-1] for p in found}
    target = "nodes" if target_label == ANY else f":{target_label} nodes"
    summary = (f"{len(found)} path(s) of 1 to {max_hops} hops lead from {start_name} "
               f"to {len(ends)} distinct {target}.")
    scope = f"{start_name} | 1..{max_hops} hops" + ("" if target_label == ANY else f" | {target_label}")
    return make_result("Multi-Hop", scope, cypher, rows, ["Hops", "End Node", "End Label", "Path"],
                       nodes, edges, summary, focus=[start_id])


def query_filtered_pattern(graph, rel_type, source_label, target_label, limit) -> dict:
    """MATCH (a:Label)-[r:TYPE]->(b:Label) RETURN a, b."""
    cypher = (
        f"MATCH {node_pattern('a', source_label)}-[r:{rel_type}]->{node_pattern('b', target_label)}\n"
        "RETURN a.name AS source, type(r) AS relationship, b.name AS target, properties(r) AS relProps\n"
        "ORDER BY source, target"
    )
    cypher = with_limit(cypher, limit)

    hits = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        if d["type"] != rel_type:
            continue
        if source_label != ANY and node_label(graph, u) != source_label:
            continue
        if target_label != ANY and node_label(graph, v) != target_label:
            continue
        hits.append((u, v, k, d))
    hits = apply_limit(sorted(hits, key=lambda h: (node_name(graph, h[0]), node_name(graph, h[1]))), limit)

    rows = [[node_name(graph, u), node_label(graph, u), d["type"], node_name(graph, v),
             node_label(graph, v), format_props(d["props"])] for u, v, k, d in hits]
    nodes = {h[0] for h in hits} | {h[1] for h in hits}
    edges = {(u, v, k) for u, v, k, _ in hits}
    src = "any node" if source_label == ANY else f":{source_label}"
    tgt = "any node" if target_label == ANY else f":{target_label}"
    summary = f"{len(hits)} {rel_type} relationship(s) match the pattern ({src})-[:{rel_type}]->({tgt})."
    scope = f"{source_label if source_label != ANY else 'Any'} -{rel_type}-> {target_label if target_label != ANY else 'Any'}"
    return make_result("Pattern Match", scope, cypher, rows,
                       ["Source", "Source Label", "Relationship", "Target", "Target Label", "Rel Properties"],
                       nodes, edges, summary)


def query_aggregation(graph, group_by, label, top_n) -> dict:
    """count() aggregations with ORDER BY ... DESC and LIMIT."""
    groups = []  # (group, count, node ids, edge ids)
    if group_by in SIMULATION_CONFIG["aggregation_modes"][:2]:
        outgoing_only = group_by == SIMULATION_CONFIG["aggregation_modes"][0]
        arrow = "->" if outgoing_only else "-"
        cypher = (
            f"MATCH {node_pattern('n', label)}-[r]{arrow}()\n"
            "RETURN n.name AS node, count(r) AS relCount\n"
            "ORDER BY relCount DESC, node ASC"
        )
        for n in sorted_node_ids(graph, label):
            edges = {(u, v, k) for u, v, k in graph.out_edges(n, keys=True)}
            if not outgoing_only:
                edges |= {(u, v, k) for u, v, k in graph.in_edges(n, keys=True)}
            if edges:
                groups.append((node_name(graph, n), len(edges), {n}, edges))
        group_col, count_col = "Node", "Relationship Count"
    elif group_by == SIMULATION_CONFIG["aggregation_modes"][2]:
        cypher = (
            "MATCH ()-[r]->()\n"
            "RETURN type(r) AS relType, count(r) AS relCount\n"
            "ORDER BY relCount DESC, relType ASC"
        )
        for rel_type in relationship_types(graph):
            edges = {(u, v, k) for u, v, k, d in graph.edges(keys=True, data=True) if d["type"] == rel_type}
            nodes = {e[0] for e in edges} | {e[1] for e in edges}
            groups.append((rel_type, len(edges), nodes, edges))
        group_col, count_col = "Relationship Type", "Relationship Count"
    else:
        cypher = (
            "MATCH (n)\n"
            "RETURN labels(n)[0] AS label, count(n) AS nodeCount\n"
            "ORDER BY nodeCount DESC, label ASC"
        )
        for lab in graph_labels(graph):
            nodes = set(sorted_node_ids(graph, lab))
            groups.append((lab, len(nodes), nodes, set()))
        group_col, count_col = "Label", "Node Count"
    cypher = with_limit(cypher, top_n)

    groups = sorted(groups, key=lambda g: (-g[1], g[0]))[:top_n]
    rows = [[i + 1, g[0], g[1]] for i, g in enumerate(groups)]
    nodes = set().union(*(g[2] for g in groups)) if groups else set()
    edges = set().union(*(g[3] for g in groups)) if groups else set()
    chart = pd.DataFrame({"group": [g[0] for g in groups], "count": [g[1] for g in groups]})
    if groups:
        summary = (f"Top {len(groups)} by {group_by.lower()}: "
                   f"{groups[0][0]} leads with {groups[0][1]}.")
    else:
        summary = f"No groups found for {group_by.lower()}."
    scope = group_by + ("" if label == ANY or group_by not in SIMULATION_CONFIG["aggregation_modes"][:2]
                        else f" | {label}")
    return make_result("Aggregation", scope, cypher, rows, ["Rank", group_col, count_col],
                       nodes, edges, summary, chart=chart)


# ======================================================================================
# 2b. QUERY INTERPRETER: RUNS QUERIES THE STUDENT EDITS BY HAND
# ======================================================================================
# A small, read-only query interpreter evaluated directly on the networkx graph:
# MATCH (several patterns / clauses, variable-length paths), WHERE, RETURN (DISTINCT, AS,
# count()), ORDER BY, SKIP and LIMIT. Relationships are not reused within one MATCH pattern.

MAX_VAR_HOPS = 5
MAX_MATCH_STEPS = 50000
WRITE_CLAUSES = {"CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "LOAD", "FOREACH"}
UNSUPPORTED_CLAUSES = {"WITH", "UNWIND", "OPTIONAL", "CALL", "UNION", "USE"}
CLAUSE_WORDS = {"MATCH", "WHERE", "RETURN", "ORDER", "SKIP", "LIMIT"} | WRITE_CLAUSES | UNSUPPORTED_CLAUSES
CYPHER_FUNCTIONS = {"type", "labels", "length", "size", "properties", "nodes", "relationships",
                    "tolower", "toupper", "id", "count"}
COMPARISON_OPS = ("=", "<>", "<", ">", "<=", ">=")

TOKEN_RE = re.compile(r"""
    (?P<space>\s+|//[^\n]*)
  | (?P<str>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<num>\d+\.\d+|\d+)
  | (?P<ident>[A-Za-z_][A-Za-z_0-9]*|`[^`]+`)
  | (?P<op><>|<=|>=|->|<-|\.\.|[-=<>()\[\]{}:,.*|;])
""", re.VERBOSE)


class CypherError(Exception):
    """A query the lab's Cypher subset cannot parse or run; the message is shown to the student."""


@dataclass(frozen=True)
class NodeRef:
    id: str


@dataclass(frozen=True)
class RelRef:
    u: str
    v: str
    k: str


@dataclass(frozen=True)
class PathRef:
    nodes: tuple  # node ids, in walking order
    rels: tuple   # RelRef per hop


def tokenize(text) -> list:
    """Splits a query into (kind, value, start, end) tokens."""
    tokens, pos = [], 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise CypherError(f"Unexpected character '{text[pos]}' at position {pos + 1}.")
        kind, value = m.lastgroup, m.group()
        if kind == "str":
            value = re.sub(r"\\(.)", r"\1", value[1:-1])
        elif kind == "ident" and value.startswith("`"):
            value = value[1:-1]
        if kind != "space":
            tokens.append((kind, value, m.start(), m.end()))
        pos = m.end()
    return tokens


class TokenStream:
    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def peek(self, offset=0):
        j = self.i + offset
        return self.tokens[j] if j < len(self.tokens) else None

    def at_end(self) -> bool:
        return self.i >= len(self.tokens)

    def next(self):
        tok = self.peek()
        if tok is None:
            raise CypherError("The query ends too early; something is missing at the end.")
        self.i += 1
        return tok

    def at(self, value, offset=0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok[0] == "op" and tok[1] == value

    def at_word(self, word, offset=0) -> bool:
        tok = self.peek(offset)
        return tok is not None and tok[0] == "ident" and tok[1].upper() == word

    def at_ident(self) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "ident"

    def expect(self, value):
        tok = self.next()
        if tok[0] != "op" or tok[1] != value:
            raise CypherError(f"Expected '{value}' but found '{tok[1]}'.")
        return tok

    def expect_ident(self, what) -> str:
        tok = self.next()
        if tok[0] != "ident":
            raise CypherError(f"Expected {what} but found '{tok[1]}'.")
        return tok[1]


def token_text(text, toks) -> str:
    return " ".join(text[toks[0][2]:toks[-1][3]].split())


def split_top(toks, sep=",") -> list:
    """Splits tokens on a separator that is not inside brackets."""
    parts, cur, depth = [], [], 0
    for tok in toks:
        if tok[0] == "op" and tok[1] in ("(", "[", "{"):
            depth += 1
        elif tok[0] == "op" and tok[1] in (")", "]", "}"):
            depth -= 1
        if depth == 0 and tok[0] == "op" and tok[1] == sep:
            parts.append(cur)
            cur = []
            continue
        cur.append(tok)
    parts.append(cur)
    if any(not p for p in parts):
        raise CypherError("There is an extra or missing comma.")
    return parts


def split_clauses(tokens) -> list:
    """Groups tokens under their clause keyword: [(MATCH, [...]), (WHERE, [...]), ...]."""
    clauses, depth, i = [], 0, 0
    while i < len(tokens):
        tok = tokens[i]
        prev = tokens[i - 1] if i else None
        if tok[0] == "op" and tok[1] in ("(", "[", "{"):
            depth += 1
        elif tok[0] == "op" and tok[1] in (")", "]", "}"):
            depth -= 1
        word = tok[1].upper() if tok[0] == "ident" else None
        is_clause = (depth == 0 and word in CLAUSE_WORDS
                     and not (prev and prev[0] == "op" and prev[1] == ".")
                     and not (word == "WITH" and prev and prev[0] == "ident"
                              and prev[1].upper() in ("STARTS", "ENDS")))
        if is_clause:
            if word == "ORDER":
                if not (i + 1 < len(tokens) and tokens[i + 1][0] == "ident" and tokens[i + 1][1].upper() == "BY"):
                    raise CypherError("Write ORDER BY, not just ORDER.")
                i += 1
            clauses.append((word, []))
        elif not clauses:
            raise CypherError("A query must start with MATCH.")
        else:
            clauses[-1][1].append(tok)
        i += 1
    return clauses


def parse_literal(ts):
    negative = ts.at("-")
    if negative:
        ts.next()
    tok = ts.next()
    if tok[0] == "num":
        value = float(tok[1]) if "." in tok[1] else int(tok[1])
        return -value if negative else value
    if not negative and tok[0] == "str":
        return tok[1]
    if not negative and tok[0] == "ident" and tok[1].lower() in ("true", "false", "null"):
        return {"true": True, "false": False, "null": None}[tok[1].lower()]
    raise CypherError(f"'{tok[1]}' is not a value. Put text in quotes, like \"The Matrix\".")


def parse_map(ts) -> dict:
    ts.expect("{")
    props = {}
    while not ts.at("}"):
        key = ts.expect_ident("a property name")
        ts.expect(":")
        props[key] = parse_literal(ts)
        if ts.at(","):
            ts.next()
        elif not ts.at("}"):
            raise CypherError("Separate properties with commas, like {name: \"X\", born: 1964}.")
    ts.expect("}")
    return props


def parse_node(ts) -> dict:
    ts.expect("(")
    var = ts.next()[1] if ts.at_ident() else None
    labels = []
    while ts.at(":"):
        ts.next()
        labels.append(ts.expect_ident("a label"))
    props = parse_map(ts) if ts.at("{") else {}
    ts.expect(")")
    return {"var": var, "labels": labels, "props": props}


def parse_hops(ts) -> tuple:
    """The part after '*': '' -> 1..max, '2' -> 2..2, '1..3', '..3', '2..'. Returns (low, high, capped)."""
    low = high = None
    if ts.peek() and ts.peek()[0] == "num":
        low = int(float(ts.next()[1]))
    if ts.at(".."):
        ts.next()
        if ts.peek() and ts.peek()[0] == "num":
            high = int(float(ts.next()[1]))
    elif low is not None:
        high = low
    low = 1 if low is None else low
    capped = high is None or high > MAX_VAR_HOPS
    high = MAX_VAR_HOPS if capped else high
    if low > high:
        raise CypherError(f"The hop range is empty (this lab allows paths of up to {MAX_VAR_HOPS} hops).")
    return low, high, capped


def parse_rel(ts) -> dict:
    if not (ts.at("<-") or ts.at("-")):
        found = ts.peek()[1] if ts.peek() else "the end"
        raise CypherError(f"Expected a relationship like -[:TYPE]-> but found '{found}'.")
    left = ts.next()[1] == "<-"
    var, types, hops, props = None, [], None, {}
    if ts.at("["):
        ts.next()
        if ts.at_ident():
            var = ts.next()[1]
        if ts.at(":"):
            ts.next()
            types.append(ts.expect_ident("a relationship type"))
            while ts.at("|"):
                ts.next()
                if ts.at(":"):
                    ts.next()
                types.append(ts.expect_ident("a relationship type"))
        if ts.at("*"):
            ts.next()
            hops = parse_hops(ts)
        if ts.at("{"):
            props = parse_map(ts)
        ts.expect("]")
    if not (ts.at("->") or ts.at("-")):
        raise CypherError("A relationship must end with - or ->.")
    right = ts.next()[1] == "->"
    if left and right:
        raise CypherError("A relationship cannot point both ways (<- ... ->).")
    direction = "in" if left else "out" if right else "both"
    return {"var": var, "types": types, "hops": hops, "props": props, "direction": direction}


def parse_pattern(toks) -> dict:
    ts = TokenStream(toks)
    path_var = None
    if ts.at_ident() and ts.at("=", 1):
        path_var = ts.next()[1]
        ts.next()
    nodes, rels = [parse_node(ts)], []
    while not ts.at_end():
        rels.append(parse_rel(ts))
        nodes.append(parse_node(ts))
    return {"path": path_var, "nodes": nodes, "rels": rels}


def parse_atom(ts):
    kind, value = ts.next()[:2]
    if kind == "op" and value == "-":
        return ("neg", parse_atom(ts))
    if kind == "op" and value == "(":
        expr = parse_or(ts)
        ts.expect(")")
        return parse_postfix(ts, expr)
    if kind == "op" and value == "[":
        items = []
        while not ts.at("]"):
            items.append(parse_or(ts))
            if ts.at(","):
                ts.next()
            elif not ts.at("]"):
                raise CypherError("Separate list items with commas, like [1999, 2010].")
        ts.expect("]")
        return ("list", items)
    if kind == "num":
        return ("lit", float(value) if "." in value else int(value))
    if kind == "str":
        return ("lit", value)
    if kind == "ident":
        low = value.lower()
        if low in ("true", "false", "null"):
            return ("lit", {"true": True, "false": False, "null": None}[low])
        if ts.at("("):
            if low not in CYPHER_FUNCTIONS:
                raise CypherError(f"The function {value}() is not supported here. "
                                  "Try type(), labels(), length(), properties() or count().")
            ts.next()
            if low == "count" and ts.at("*"):
                ts.next()
                ts.expect(")")
                return ("count", None, False)
            distinct = ts.at_word("DISTINCT")
            if distinct:
                ts.next()
            arg = parse_or(ts)
            ts.expect(")")
            if low == "count":
                return ("count", arg, distinct)
            return parse_postfix(ts, ("func", low, arg))
        return parse_postfix(ts, ("var", value))
    raise CypherError(f"Did not understand '{value}' here.")


def parse_postfix(ts, expr):
    while True:
        if ts.at("."):
            ts.next()
            expr = ("prop", expr, ts.expect_ident("a property name"))
        elif ts.at("["):
            ts.next()
            index = parse_or(ts)
            ts.expect("]")
            expr = ("index", expr, index)
        else:
            return expr


def parse_comparison(ts):
    left = parse_atom(ts)
    tok = ts.peek()
    if tok is not None and tok[0] == "op" and tok[1] in COMPARISON_OPS:
        ts.next()
        return ("cmp", tok[1], left, parse_atom(ts))
    for word in ("CONTAINS", "IN"):
        if ts.at_word(word):
            ts.next()
            return ("cmp", word, left, parse_atom(ts))
    if ts.at_word("STARTS") or ts.at_word("ENDS"):
        word = ts.next()[1].upper()
        if not ts.at_word("WITH"):
            raise CypherError(f"Write {word} WITH.")
        ts.next()
        return ("cmp", f"{word} WITH", left, parse_atom(ts))
    if ts.at_word("IS"):
        ts.next()
        negate = ts.at_word("NOT")
        if negate:
            ts.next()
        if not ts.at_word("NULL"):
            raise CypherError("Write IS NULL or IS NOT NULL.")
        ts.next()
        return ("isnull", left, negate)
    return left


def parse_not(ts):
    if ts.at_word("NOT"):
        ts.next()
        return ("not", parse_not(ts))
    return parse_comparison(ts)


def parse_and(ts):
    left = parse_not(ts)
    while ts.at_word("AND"):
        ts.next()
        left = ("and", left, parse_not(ts))
    return left


def parse_or(ts):
    left = parse_and(ts)
    while ts.at_word("OR") or ts.at_word("XOR"):
        op = ts.next()[1].lower()
        left = (op, left, parse_and(ts))
    return left


def parse_expression(toks):
    if not toks:
        raise CypherError("Something is missing after WHERE, RETURN or ORDER BY.")
    ts = TokenStream(toks)
    expr = parse_or(ts)
    if not ts.at_end():
        raise CypherError(f"Did not understand '{ts.peek()[1]}' here.")
    return expr


def expr_children(expr) -> list:
    return [part for part in expr[1:] if isinstance(part, tuple)] + \
        [item for part in expr[1:] if isinstance(part, list) for item in part]


def contains_count(expr) -> bool:
    return expr[0] == "count" or any(contains_count(child) for child in expr_children(expr))


def expr_vars(expr) -> set:
    found = {expr[1]} if expr[0] == "var" else set()
    for child in expr_children(expr):
        found |= expr_vars(child)
    return found


def parse_count_clause(toks, word) -> int:
    if len(toks) != 1 or toks[0][0] != "num" or "." in toks[0][1]:
        raise CypherError(f"{word} needs a whole number, like {word} 5.")
    return int(toks[0][1])


def parse_return(toks, text) -> dict:
    distinct = bool(toks) and toks[0][0] == "ident" and toks[0][1].upper() == "DISTINCT"
    if distinct:
        toks = toks[1:]
    if not toks:
        raise CypherError("RETURN needs something to show, like RETURN n.name.")
    if len(toks) == 1 and toks[0][0] == "op" and toks[0][1] == "*":
        return {"distinct": distinct, "items": None}
    items = []
    for part in split_top(toks):
        alias = None
        if len(part) >= 3 and part[-2][0] == "ident" and part[-2][1].upper() == "AS" and part[-1][0] == "ident":
            alias, part = part[-1][1], part[:-2]
        expr = parse_expression(part)
        if contains_count(expr) and expr[0] != "count":
            raise CypherError("Use count() on its own in a RETURN item, like count(r) AS total.")
        expr_text = token_text(text, part)
        items.append({"name": alias or expr_text, "text": expr_text, "expr": expr, "agg": expr[0] == "count"})
    return {"distinct": distinct, "items": items}


def parse_order(toks, text) -> list:
    specs = []
    for part in split_top(toks):
        desc = False
        if part[-1][0] == "ident" and part[-1][1].upper() in ("ASC", "ASCENDING", "DESC", "DESCENDING"):
            desc = part[-1][1].upper().startswith("DESC")
            part = part[:-1]
        if not part:
            raise CypherError("ORDER BY needs something to sort by.")
        specs.append({"text": token_text(text, part), "expr": parse_expression(part), "desc": desc})
    return specs


def parse_query(text) -> dict:
    tokens = tokenize(text)
    while tokens and tokens[-1][0] == "op" and tokens[-1][1] == ";":
        tokens.pop()
    if not tokens:
        raise CypherError("The query is empty.")
    if any(tok[0] == "op" and tok[1] == ";" for tok in tokens):
        raise CypherError("Run one query at a time (remove the ';' in the middle).")
    clauses = split_clauses(tokens)
    if clauses[0][0] not in {"MATCH"} | WRITE_CLAUSES | UNSUPPORTED_CLAUSES:
        raise CypherError("A query must start with MATCH.")
    query = {"matches": [], "where": [], "return": None, "order": [], "skip": 0, "limit": None}
    stage = "match"
    for word, toks in clauses:
        if word in WRITE_CLAUSES:
            raise CypherError(f"{word} changes the graph. This lab is read-only, so only MATCH queries can run.")
        if word in UNSUPPORTED_CLAUSES:
            raise CypherError(f"{word} is not supported in this lab. Use MATCH, WHERE, RETURN, ORDER BY and LIMIT.")
        if word == "MATCH":
            if stage != "match":
                raise CypherError("MATCH must come before RETURN.")
            if not toks:
                raise CypherError("MATCH needs a pattern, like (n:Person).")
            query["matches"].append([parse_pattern(part) for part in split_top(toks)])
        elif word == "WHERE":
            if stage != "match" or not query["matches"]:
                raise CypherError("WHERE must come right after a MATCH.")
            expr = parse_expression(toks)
            if contains_count(expr):
                raise CypherError("count() can only be used in RETURN, not in WHERE.")
            query["where"].append(expr)
        elif word == "RETURN":
            if stage != "match" or not query["matches"]:
                raise CypherError("RETURN must come after MATCH.")
            query["return"] = parse_return(toks, text)
            stage = "return"
        elif word == "ORDER":
            if stage != "return":
                raise CypherError("ORDER BY must come after RETURN.")
            query["order"] = parse_order(toks, text)
            stage = "order"
        elif word == "SKIP":
            if stage not in ("return", "order"):
                raise CypherError("SKIP must come after RETURN (and ORDER BY).")
            query["skip"] = parse_count_clause(toks, "SKIP")
            stage = "skip"
        elif word == "LIMIT":
            if stage not in ("return", "order", "skip"):
                raise CypherError("LIMIT must come last, after RETURN.")
            query["limit"] = parse_count_clause(toks, "LIMIT")
            stage = "limit"
    if query["return"] is None:
        raise CypherError("Add a RETURN clause to say what to show, like RETURN n.name.")
    return query


def pattern_vars(query) -> list:
    """User-named variables in the order they appear in the MATCH patterns."""
    names = []
    for patterns in query["matches"]:
        for pat in patterns:
            candidates = [pat["path"]] + [n["var"] for n in pat["nodes"]] + [r["var"] for r in pat["rels"]]
            names += [v for v in candidates if v and v not in names]
    return names


def check_variables(query, declared):
    known = set(declared)
    items = query["return"]["items"] or []
    exprs = list(query["where"]) + [it["expr"] for it in items]
    columns = {it["name"] for it in items} | {it["text"] for it in items}
    exprs += [spec["expr"] for spec in query["order"] if spec["text"] not in columns]
    for expr in exprs:
        unknown = sorted(expr_vars(expr) - known)
        if unknown:
            raise CypherError(f"Unknown variable '{unknown[0]}'. Variables come from the MATCH pattern, "
                              "like p in (p:Person).")


def cypher_compare(op, a, b):
    """Cypher comparison with null: any comparison involving null is null (neither true nor false)."""
    if op == "IN":
        if b is None or a is None:
            return None
        if not isinstance(b, (list, tuple)):
            raise CypherError("IN needs a list on the right, like n.released IN [1999, 2010].")
        return a in b
    if a is None or b is None:
        return None
    if op in ("=", "<>"):
        return (a == b) == (op == "=")
    if op in ("CONTAINS", "STARTS WITH", "ENDS WITH"):
        if not (isinstance(a, str) and isinstance(b, str)):
            return None
        return {"CONTAINS": b in a, "STARTS WITH": a.startswith(b), "ENDS WITH": a.endswith(b)}[op]

    def is_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    if (is_num(a) and is_num(b)) or (isinstance(a, str) and isinstance(b, str)):
        return {">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]
    return None


def cypher_logic(op, a, b):
    for v in (a, b):
        if v is not None and not isinstance(v, bool):
            raise CypherError(f"{op.upper()} needs true/false conditions on both sides.")
    if op == "and":
        if a is False or b is False:
            return False
        return None if a is None or b is None else True
    if op == "or":
        if a is True or b is True:
            return True
        return None if a is None or b is None else False
    return None if a is None or b is None else a != b


def call_function(graph, name, value):
    if value is None:
        return None
    if name == "type" and isinstance(value, RelRef):
        return graph.edges[value.u, value.v, value.k]["type"]
    if name == "labels" and isinstance(value, NodeRef):
        return [graph.nodes[value.id]["label"]]
    if name == "id" and isinstance(value, NodeRef):
        return value.id
    if name == "id" and isinstance(value, RelRef):
        return value.k
    if name == "properties" and isinstance(value, NodeRef):
        return dict(graph.nodes[value.id]["props"])
    if name == "properties" and isinstance(value, RelRef):
        return dict(graph.edges[value.u, value.v, value.k]["props"])
    if name == "length" and isinstance(value, PathRef):
        return len(value.rels)
    if name in ("length", "size") and isinstance(value, (list, tuple, str)):
        return len(value)
    if name == "nodes" and isinstance(value, PathRef):
        return [NodeRef(n) for n in value.nodes]
    if name == "relationships" and isinstance(value, PathRef):
        return list(value.rels)
    if name in ("tolower", "toupper") and isinstance(value, str):
        return value.lower() if name == "tolower" else value.upper()
    raise CypherError(f"{name}() cannot be used on this kind of value.")


def eval_expr(graph, expr, row):
    kind = expr[0]
    if kind == "lit":
        return expr[1]
    if kind == "var":
        if expr[1] not in row:
            raise CypherError(f"Unknown variable '{expr[1]}'.")
        return row[expr[1]]
    if kind == "prop":
        base = eval_expr(graph, expr[1], row)
        if base is None:
            return None
        if isinstance(base, NodeRef):
            return graph.nodes[base.id]["props"].get(expr[2])
        if isinstance(base, RelRef):
            return graph.edges[base.u, base.v, base.k]["props"].get(expr[2])
        if isinstance(base, dict):
            return base.get(expr[2])
        raise CypherError(f"Only nodes and relationships have properties (.{expr[2]}).")
    if kind == "index":
        base, index = eval_expr(graph, expr[1], row), eval_expr(graph, expr[2], row)
        if base is None or index is None:
            return None
        if not isinstance(base, (list, tuple)):
            raise CypherError("Only lists can be indexed with [ ].")
        if not isinstance(index, int) or isinstance(index, bool):
            raise CypherError("A list index must be a whole number, like labels(n)[0].")
        return base[index] if -len(base) <= index < len(base) else None
    if kind == "list":
        return [eval_expr(graph, item, row) for item in expr[1]]
    if kind == "neg":
        value = eval_expr(graph, expr[1], row)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            raise CypherError("A minus sign only works on numbers.")
        return None if value is None else -value
    if kind == "func":
        return call_function(graph, expr[1], eval_expr(graph, expr[2], row))
    if kind == "count":
        raise CypherError("count() can only be used in RETURN.")
    if kind == "not":
        value = eval_expr(graph, expr[1], row)
        if value is not None and not isinstance(value, bool):
            raise CypherError("NOT needs a true/false condition.")
        return None if value is None else not value
    if kind in ("and", "or", "xor"):
        return cypher_logic(kind, eval_expr(graph, expr[1], row), eval_expr(graph, expr[2], row))
    if kind == "isnull":
        is_null = eval_expr(graph, expr[1], row) is None
        return not is_null if expr[2] else is_null
    return cypher_compare(expr[1], eval_expr(graph, expr[2], row), eval_expr(graph, expr[3], row))


def node_matches(graph, node_id, pat) -> bool:
    data = graph.nodes[node_id]
    if any(label != data["label"] for label in pat["labels"]):
        return False
    return all(data["props"].get(k) == v for k, v in pat["props"].items())


def rel_matches(data, pat) -> bool:
    if pat["types"] and data["type"] not in pat["types"]:
        return False
    return all(data["props"].get(k) == v for k, v in pat["props"].items())


def bind(row, var, value):
    """Adds var=value to a row; returns None when var is already bound to something else."""
    if row is None or not var:
        return row
    if var in row:
        return row if row[var] == value else None
    return {**row, var: value}


def rel_steps(graph, node_id, direction):
    if direction in ("out", "both"):
        for u, v, k, d in graph.out_edges(node_id, keys=True, data=True):
            yield v, RelRef(u, v, k), d
    if direction in ("in", "both"):
        for u, v, k, d in graph.in_edges(node_id, keys=True, data=True):
            yield u, RelRef(u, v, k), d


def match_clause(graph, patterns, start_row, budget) -> list:
    """All ways one MATCH clause (its comma-separated patterns) fits the graph, extending start_row."""
    results = []

    def spend():
        budget[0] -= 1
        if budget[0] < 0:
            raise CypherError("This query explores too many paths for the lab. "
                              "Add labels, a relationship type or fewer hops.")

    def bind_node(row, pat, node_id):
        return bind(row, pat["var"], NodeRef(node_id)) if node_matches(graph, node_id, pat) else None

    def match_from(pi, row, used):
        if pi == len(patterns):
            results.append(row)
            return
        first = patterns[pi]["nodes"][0]
        bound = row.get(first["var"]) if first["var"] else None
        starts = [bound.id] if isinstance(bound, NodeRef) else sorted_node_ids(graph)
        for start in starts:
            new_row = bind_node(row, first, start)
            if new_row is not None:
                walk(pi, 0, start, new_row, used, [start], [])

    def walk(pi, ri, cur, row, used, path_nodes, path_rels):
        spend()
        pat = patterns[pi]
        if ri == len(pat["rels"]):
            row = {**row, "#nodes": row.get("#nodes", ()) + tuple(path_nodes),
                   "#rels": row.get("#rels", ()) + tuple(path_rels)}
            row.setdefault("#start", path_nodes[0])
            if pat["path"]:
                row = bind(row, pat["path"], PathRef(tuple(path_nodes), tuple(path_rels)))
            if row is not None:
                match_from(pi + 1, row, used)
            return
        rel, nxt = pat["rels"][ri], pat["nodes"][ri + 1]
        if rel["hops"] is None:
            for other, ref, data in rel_steps(graph, cur, rel["direction"]):
                if ref in used or not rel_matches(data, rel):
                    continue
                new_row = bind_node(bind(row, rel["var"], ref), nxt, other)
                if new_row is not None:
                    walk(pi, ri + 1, other, new_row, used | {ref}, path_nodes + [other], path_rels + [ref])
            return
        low, high, _ = rel["hops"]

        def extend(node, hops, hop_nodes, used_now):
            spend()
            if len(hops) >= low:
                new_row = bind_node(bind(row, rel["var"], tuple(hops)), nxt, node)
                if new_row is not None:
                    walk(pi, ri + 1, node, new_row, used_now, path_nodes + hop_nodes, path_rels + hops)
            if len(hops) == high:
                return
            for other, ref, data in rel_steps(graph, node, rel["direction"]):
                if ref not in used_now and rel_matches(data, rel):
                    extend(other, hops + [ref], hop_nodes + [other], used_now | {ref})

        extend(cur, [], [], used)

    match_from(0, start_row, frozenset())
    return results


def hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, hashable(v)) for k, v in value.items()))
    return value


def display_value(graph, value):
    """Shows a Cypher value the way Neo4j prints it: (:Label {...}), [:TYPE {...}], paths, lists."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, NodeRef):
        data = graph.nodes[value.id]
        return f"(:{data['label']} {format_props(data['props'])})"
    if isinstance(value, RelRef):
        data = graph.edges[value.u, value.v, value.k]
        props = format_props(data["props"])
        return f"[:{data['type']}{' ' + props if props else ''}]"
    if isinstance(value, PathRef):
        parts = [node_name(graph, value.nodes[0])]
        for rel, prev, nxt in zip(value.rels, value.nodes, value.nodes[1:]):
            rel_type = graph.edges[rel.u, rel.v, rel.k]["type"]
            parts.append(f" -[:{rel_type}]-> " if rel.u == prev else f" <-[:{rel_type}]- ")
            parts.append(node_name(graph, nxt))
        return "".join(parts)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(display_value(graph, v)) for v in value) + "]"
    if isinstance(value, dict):
        return format_props(value) or "{}"
    return str(value)


def sort_key(graph, value):
    """Cypher ordering: nulls last when ascending (so first when descending)."""
    if value is None:
        return (1,)
    if isinstance(value, bool):
        return (0, 0, value)
    if isinstance(value, (int, float)):
        return (0, 1, value)
    if isinstance(value, str):
        return (0, 2, value)
    return (0, 3, str(display_value(graph, value)))


def project_rows(graph, items, rows, distinct) -> list:
    """RETURN: one (values, source rows) pair per output row; count() groups by the other items."""
    if any(it["agg"] for it in items):
        groups = {}
        for row in rows:
            values = [None if it["agg"] else eval_expr(graph, it["expr"], row) for it in items]
            key = hashable([v for it, v in zip(items, values) if not it["agg"]])
            groups.setdefault(key, (values, []))[1].append(row)
        if not groups and all(it["agg"] for it in items):
            groups[()] = ([None] * len(items), [])
        out = []
        for values, members in groups.values():
            final = []
            for it, value in zip(items, values):
                if not it["agg"]:
                    final.append(value)
                    continue
                _, arg, count_distinct = it["expr"]
                if arg is None:
                    final.append(len(members))
                    continue
                counted = [v for v in (eval_expr(graph, arg, m) for m in members) if v is not None]
                final.append(len({hashable(v) for v in counted}) if count_distinct else len(counted))
            out.append((final, members))
    else:
        out = [([eval_expr(graph, it["expr"], row) for it in items], [row]) for row in rows]
    if distinct:
        merged = {}
        for values, members in out:
            merged.setdefault(hashable(values), (values, []))[1].extend(members)
        out = list(merged.values())
    return out


def order_rows(graph, specs, items, out) -> list:
    columns = {}
    for i, it in enumerate(items):
        columns.setdefault(it["name"], i)
        columns.setdefault(it["text"], i)
    aggregated = any(it["agg"] for it in items)
    for spec in reversed(specs):  # stable sorts, last key first
        index = columns.get(spec["text"])
        if index is not None:
            out = sorted(out, key=lambda r, i=index: sort_key(graph, r[0][i]), reverse=spec["desc"])
        elif aggregated:
            raise CypherError(f"ORDER BY {spec['text']}: when counting, sort by a column you RETURN.")
        else:
            out = sorted(out, key=lambda r, e=spec["expr"]: sort_key(graph, eval_expr(graph, e, r[1][0])),
                         reverse=spec["desc"])
    return out


def query_custom(graph, text) -> dict:
    """Parses and runs a hand-edited Cypher query; raises CypherError with a student-friendly message."""
    query = parse_query(text)
    declared = pattern_vars(query)
    check_variables(query, declared)

    budget = [MAX_MATCH_STEPS]
    rows = [{}]
    for patterns in query["matches"]:
        rows = [new for row in rows for new in match_clause(graph, patterns, row, budget)]
    rows = [row for row in rows if all(eval_expr(graph, w, row) is True for w in query["where"])]

    items = query["return"]["items"]
    if items is None:  # RETURN *
        if not declared:
            raise CypherError("RETURN * needs named variables in the pattern, like (p:Person).")
        items = [{"name": v, "text": v, "expr": ("var", v), "agg": False} for v in declared]
    out = project_rows(graph, items, rows, query["return"]["distinct"])
    out = order_rows(graph, query["order"], items, out)[query["skip"]:]
    if query["limit"] is not None:
        out = out[:query["limit"]]

    columns = []
    for it in items:
        name, n = it["name"], 2
        while name in columns:
            name, n = f"{it['name']} ({n})", n + 1
        columns.append(name)
    table_rows = [[display_value(graph, v) for v in values] for values, _ in out]
    for c in range(len(columns)):  # a column mixing numbers and text is shown as text
        if len({type(row[c]) for row in table_rows}) > 1:
            for row in table_rows:
                row[c] = str(row[c])

    nodes, edges, starts = set(), set(), set()
    for _, members in out:
        for row in members:
            nodes.update(row.get("#nodes", ()))
            edges.update((r.u, r.v, r.k) for r in row.get("#rels", ()))
            if "#start" in row:
                starts.add(row["#start"])

    chart = None
    if len(items) == 2 and [it["agg"] for it in items] == [False, True] and table_rows:
        chart = pd.DataFrame({"group": [str(r[0]) for r in table_rows], "count": [r[1] for r in table_rows]})

    summary = (f"Your query returned {len(table_rows)} row(s), touching {len(nodes)} node(s) "
               f"and {len(edges)} relationship(s).")
    if any(rel["hops"] and rel["hops"][2] for patterns in query["matches"]
           for pat in patterns for rel in pat["rels"]):
        summary += f" Open-ended hop ranges are capped at {MAX_VAR_HOPS} hops in this lab."
    one_line = " ".join(text.split())
    scope = one_line if len(one_line) <= 60 else one_line[:57] + "..."
    return make_result("Custom Cypher", scope, text.strip(), table_rows, columns, nodes, edges, summary,
                       focus=starts if len(starts) == 1 else None, chart=chart)


# ======================================================================================
# 3. GRAPH VISUALIZATION (PLOTLY)
# ======================================================================================

def build_graph_figure(graph, pos, highlight_nodes=None, highlight_edges=None,
                       focus_nodes=None, show_edge_labels=True) -> go.Figure:
    """Draws the knowledge graph; when highlight sets are given, matched items are emphasised."""
    cfg = SIMULATION_CONFIG
    highlighting = highlight_nodes is not None
    highlight_nodes = set(highlight_nodes or [])
    highlight_edges = set(highlight_edges or [])
    focus_nodes = set(focus_nodes or [])
    fig = go.Figure()

    base_x, base_y, hot_x, hot_y = [], [], [], []
    mid_x, mid_y, mid_text, mid_hover = [], [], [], []
    annotations = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        hot = (u, v, k) in highlight_edges
        start, end = np.array(pos[u]), np.array(pos[v])
        xs, ys = (hot_x, hot_y) if hot else (base_x, base_y)
        xs.extend([start[0], end[0], None])
        ys.extend([start[1], end[1], None])

        unit = (end - start) / (float(np.linalg.norm(end - start)) or 1.0)
        tip = end - unit * cfg["arrow_gap"]
        tail = tip - unit * cfg["arrow_length"]
        if hot:
            color = cfg["highlight_color"]
        else:
            color = cfg["faded_edge_color"] if highlighting else cfg["edge_color"]
        annotations.append(dict(
            x=tip[0], y=tip[1], ax=tail[0], ay=tail[1],
            xref="x", yref="y", axref="x", ayref="y", text="",
            showarrow=True, arrowhead=2, arrowsize=1.1,
            arrowwidth=2.5 if hot else 1.2, arrowcolor=color
        ))

        if hot or not highlighting:
            mid = (start + end) / 2
            mid_x.append(mid[0])
            mid_y.append(mid[1])
            mid_text.append(data["type"] if show_edge_labels else "")
            mid_hover.append(edge_text(graph, u, v, data))

    fig.add_trace(go.Scatter(
        x=base_x, y=base_y, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(width=1.2, color=cfg["faded_edge_color"] if highlighting else cfg["edge_color"])
    ))
    if highlighting:
        fig.add_trace(go.Scatter(
            x=hot_x if hot_x else [None], y=hot_y if hot_y else [None], mode="lines",
            hoverinfo="skip", name="Matched relationship",
            line=dict(width=3, color=cfg["highlight_color"])
        ))
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="markers+text", text=mid_text, hovertext=mid_hover,
        hoverinfo="text", textfont=dict(size=9), showlegend=False,
        marker=dict(size=14, opacity=0)
    ))

    for label, style in cfg["label_styles"].items():
        ids = sorted_node_ids(graph, label)
        if not ids:
            continue
        matched = [(not highlighting) or n in highlight_nodes for n in ids]
        hover = []
        for n in ids:
            props = graph.nodes[n]["props"]
            extra = "".join(f"<br>{k}: {v}" for k, v in props.items() if k != "name")
            hover.append(f"<b>{props['name']}</b><br>:{label}{extra}")
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids],
            y=[pos[n][1] for n in ids],
            mode="markers+text",
            name=label,
            text=[node_name(graph, n) if m else "" for n, m in zip(ids, matched)],
            textposition="bottom center",
            textfont=dict(size=11),
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                symbol=style["symbol"],
                color=style["color"],
                size=[cfg["focus_size"] if n in focus_nodes else cfg["node_size"] for n in ids],
                opacity=[1.0 if m else cfg["faded_opacity"] for m in matched],
                line=dict(
                    width=[4 if n in focus_nodes else (2.5 if highlighting and m else 0)
                           for n, m in zip(ids, matched)],
                    color=[cfg["highlight_color"] if highlighting and m else style["color"] for m in matched]
                )
            )
        ))

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig.update_layout(
        annotations=annotations,
        height=cfg["graph_height"],
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis=dict(visible=False, range=[min(xs) - 0.15, max(xs) + 0.15]),
        yaxis=dict(visible=False, range=[min(ys) - 0.18, max(ys) + 0.12]),
    )
    return fig


def build_aggregation_chart(chart_df: pd.DataFrame, group_label: str, count_label: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=chart_df["count"],
        y=chart_df["group"],
        orientation="h",
        marker=dict(color=SIMULATION_CONFIG["bar_color"]),
        text=chart_df["count"],
        textposition="outside",
        hovertemplate=f"{group_label}: %{{y}}<br>{count_label}: %{{x}}<extra></extra>"
    ))
    fig.update_layout(
        title=f"{count_label} by {group_label}",
        xaxis_title=count_label,
        yaxis=dict(autorange="reversed"),
        height=max(260, 48 * len(chart_df) + 110),
        margin=dict(l=20, r=40, t=40, b=20),
        bargap=0.35
    )
    return fig


# ======================================================================================
# 4. LAB REPORT PDF EXPORTER
# ======================================================================================

# Long free-text trial columns are listed below the trials table instead of being truncated in it.
PDF_LONG_TEXT_COLUMNS = ["Cypher Query", "Summary"]


def pdf_safe(text) -> str:
    """Core PDF fonts are Latin-1 only; map common Unicode punctuation and drop anything else."""
    replacements = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"',
                    "”": '"', "→": "->", "←": "<-", "…": "..."}
    text = str(text)
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


class LabReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | KGIRS Virtual Lab - {EXPERIMENT_CONFIG['experiment_no']} Report",
                  align="C")


def generate_pdf_report(student_name: str, student_id: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str) -> bytes:
    """Compiles experiment benchmark records into a proper, formatted PDF report document."""
    pdf = LabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Document Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, pdf_safe(f"{EXPERIMENT_CONFIG['experiment_no']}: {EXPERIMENT_CONFIG['title']}"),
             align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student & Session Info Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 22, 190, 22, "FD")

    pdf.set_xy(14, 24)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, pdf_safe(student_name or "N/A"), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Student ID / Roll:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, pdf_safe(student_id or "N/A"), 1)

    pdf.set_xy(14, 32)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(38, 5, "Experiment Date:", 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(57, 5, pdf_safe(date_str or datetime.now().strftime("%Y-%m-%d")), 0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(35, 5, "Quiz Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 9)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({int((quiz_score/quiz_total)*100 if quiz_total else 0)}%)", 1)

    pdf.ln(12)

    # 1. Objectives
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "1. Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    for obj in EXPERIMENT_CONFIG["objectives"]:
        clean_obj = pdf_safe(str(obj).replace("$", "").replace("\\", ""))
        pdf.cell(5, 5, "-", 0)
        pdf.multi_cell(0, 5, f" {clean_obj}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 2. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "2. Recorded Query Trials & Results", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No query trials recorded during this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)

        cols = [c for c in trials_df.columns if c not in PDF_LONG_TEXT_COLUMNS]
        num_cols = len(cols)
        col_w = max(18, int(190 / max(1, num_cols)))
        max_chars = int(col_w / 1.6)

        for c in cols:
            pdf.cell(col_w, 6, pdf_safe(c)[:max_chars], 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 8)
        fill = False

        for _, row in trials_df.iterrows():
            for c in cols:
                val = row[c]
                val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 5, pdf_safe(val_str)[:max_chars], 1, 0, "C", fill)
            pdf.ln()
            fill = not fill

        # Full Cypher text and result summary for each trial
        long_cols = [c for c in PDF_LONG_TEXT_COLUMNS if c in trials_df.columns]
        if long_cols:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 58, 138)
            pdf.cell(0, 6, "Executed Cypher Queries", new_x="LMARGIN", new_y="NEXT")
            for _, row in trials_df.iterrows():
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(51, 65, 85)
                pdf.cell(0, 5, pdf_safe(f"Trial {row.get('Trial #', '')} - {row.get('Query Mode', '')}"),
                         new_x="LMARGIN", new_y="NEXT")
                if "Cypher Query" in long_cols:
                    pdf.set_font("Courier", "", 8)
                    pdf.set_text_color(15, 23, 42)
                    pdf.multi_cell(0, 4, pdf_safe(row["Cypher Query"]), new_x="LMARGIN", new_y="NEXT")
                if "Summary" in long_cols:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(71, 85, 105)
                    pdf.multi_cell(0, 4, pdf_safe(f"Result: {row['Summary']}"), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
    pdf.ln(5)

    # 3. Discussion & Notes
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 7, "3. Observations & Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    notes_text = student_notes.strip() if student_notes.strip() else DEFAULT_NOTES
    pdf.multi_cell(0, 5, pdf_safe(notes_text))
    pdf.ln(8)

    # Sign-off line
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 15, 190, pdf.get_y() + 15)
    pdf.set_xy(130, pdf.get_y() + 17)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, "Instructor / Student Signature", align="C")

    return bytes(pdf.output())


DEFAULT_NOTES = (
    "Node lookups with inline properties and WHERE filters returned exactly the nodes that satisfied the "
    "conditions. Changing relationship direction changed the 1-hop results, confirming that Cypher "
    "relationships are directed. Increasing the hop range in multi-hop traversal grew the number of reachable "
    "nodes and paths quickly, and aggregation with count(), ORDER BY and LIMIT identified the most connected "
    "entities in the knowledge graph."
)


# ======================================================================================
# 5. SECTION RENDERERS: THEORY, SIMULATION, QUIZ, REPORT
# ======================================================================================

def render_theory_section():
    """Renders Section 1: Theory, Background, Objectives, and Procedure."""
    st.header("Theoretical Framework & Background")
    st.markdown(THEORY_CONTENT["background"])

    st.subheader("Learning Objectives")
    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        st.write(f"- **Goal {i+1}**: {obj}")

    st.divider()
    st.subheader("Experimental Procedure")
    for step in THEORY_CONTENT["procedure"]:
        st.write(f"- {step}")

    st.divider()
    with st.expander("Key Terminology Reference"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term", "Definition & Role"]
        )
        st.table(var_df)


def render_graph_explorer(graph, pos):
    """Sample graph overview: full visualization, node/relationship lists, and schema."""
    st.subheader("Explore the Sample Knowledge Graph")
    st.caption(GRAPH_DATA["domain"] + ". Hover over nodes and relationship labels for details; drag to pan.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Nodes", graph.number_of_nodes())
    with m2:
        st.metric("Relationships", graph.number_of_edges())
    with m3:
        st.metric("Node Labels", len(graph_labels(graph)))
    with m4:
        st.metric("Relationship Types", len(relationship_types(graph)))

    tab_graph, tab_nodes, tab_rels, tab_schema = st.tabs(
        ["Graph View", "Node List", "Relationship List", "Schema"]
    )
    with tab_graph:
        show_labels = st.checkbox("Show relationship type labels", value=True, key="explorer_edge_labels")
        st.plotly_chart(build_graph_figure(graph, pos, show_edge_labels=show_labels),
                        key="full_graph_chart")
    with tab_nodes:
        node_rows = []
        for n in sorted_node_ids(graph):
            props = graph.nodes[n]["props"]
            node_rows.append([n, node_label(graph, n), props["name"],
                              format_props({k: v for k, v in props.items() if k != "name"})])
        st.dataframe(pd.DataFrame(node_rows, columns=["ID", "Label", "name", "Other Properties"]),
                     width="stretch", hide_index=True)
    with tab_rels:
        rel_rows = [[node_name(graph, u), d["type"], node_name(graph, v), format_props(d["props"])]
                    for u, v, d in graph.edges(data=True)]
        st.dataframe(pd.DataFrame(rel_rows, columns=["Start Node", "Type", "End Node", "Properties"]),
                     width="stretch", hide_index=True)
    with tab_schema:
        label_rows = [[f":{lab}", len(sorted_node_ids(graph, lab)), ", ".join(property_keys(graph, lab))]
                      for lab in graph_labels(graph)]
        st.markdown("**Node labels**")
        st.dataframe(pd.DataFrame(label_rows, columns=["Label", "Nodes", "Properties"]),
                     width="stretch", hide_index=True)
        patterns = {}
        for u, v, d in graph.edges(data=True):
            key = f"(:{node_label(graph, u)})-[:{d['type']}]->(:{node_label(graph, v)})"
            patterns[key] = patterns.get(key, 0) + 1
        st.markdown("**Relationship patterns**")
        st.dataframe(pd.DataFrame(sorted(patterns.items()), columns=["Pattern", "Count"]),
                     width="stretch", hide_index=True)


def limit_input(container) -> int:
    with container:
        return int(st.number_input("LIMIT (0 = no limit)", min_value=0,
                                   max_value=SIMULATION_CONFIG["max_limit"], value=0, step=1))


def render_query_builder(graph) -> dict:
    """Guided Query Builder: collects options for the selected mode and executes the query."""
    cfg = SIMULATION_CONFIG
    st.subheader("Cypher Query Builder")
    mode = st.selectbox("Query pattern mode", options=list(cfg["query_modes"]), index=0)
    labels = graph_labels(graph)
    rel_types = relationship_types(graph)
    node_ids = sorted_node_ids(graph)
    default_start = node_ids.index(GRAPH_DATA["nodes"][0]["id"])

    def fmt(n):
        return node_display(graph, n)

    if mode == "Node Lookup":
        st.caption("Find nodes by label and, optionally, a property condition.")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            label = st.selectbox("Node label", [ANY] + labels, index=1)
        with c2:
            prop_key = st.selectbox("Property", [NO_FILTER] + property_keys(graph, label), index=1)
        operator, value = "=", None
        if prop_key != NO_FILTER:
            values = property_values(graph, label, prop_key)
            numeric = all(isinstance(v, (int, float)) for v in values)
            with c3:
                operator = st.selectbox("Operator", NUMERIC_OPERATORS if numeric else STRING_OPERATORS)
            with c4:
                if numeric:
                    value = int(st.number_input(f"Value of {prop_key}", value=int(values[len(values) // 2]),
                                                step=1))
                elif operator in ("=", "<>"):
                    value = st.selectbox(f"Value of {prop_key}", values)
                else:
                    value = st.text_input(f"Text for {prop_key}", value="")
        limit = limit_input(c5)
        return query_node_lookup(graph, label, prop_key, operator, value, limit)

    if mode == "Relationship Traversal (1-hop)":
        st.caption("Follow relationships one hop away from a chosen start node.")
        c1, c2, c3, c4 = st.columns([2, 1.3, 2, 1])
        with c1:
            start_id = st.selectbox("Start node (a)", node_ids, index=default_start, format_func=fmt)
        with c2:
            rel_type = st.selectbox("Relationship type", [ANY] + rel_types)
        with c3:
            direction = st.radio("Direction", list(DIRECTIONS), horizontal=False)
        limit = limit_input(c4)
        return query_one_hop(graph, start_id, rel_type, DIRECTIONS[direction], limit)

    if mode == "Multi-hop Traversal":
        st.caption("Find every path of 1 up to N hops (any relationship type, any direction) from a start node.")
        c1, c2, c3, c4 = st.columns([2, 1, 1.3, 1])
        with c1:
            start_id = st.selectbox("Start node (a)", node_ids, index=default_start, format_func=fmt)
        with c2:
            max_hops = st.radio("Maximum hops", cfg["hop_options"], horizontal=True)
        with c3:
            target_label = st.selectbox("End node label (b)", [ANY] + labels)
        limit = limit_input(c4)
        return query_multi_hop(graph, start_id, int(max_hops), target_label, limit)

    if mode == "Filtered Pattern Match":
        st.caption("Match every relationship of one type whose end nodes carry the chosen labels.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            source_label = st.selectbox("Start node label (a)", [ANY] + labels)
        with c2:
            rel_type = st.selectbox("Relationship type", rel_types)
        with c3:
            target_label = st.selectbox("End node label (b)", [ANY] + labels)
        limit = limit_input(c4)
        return query_filtered_pattern(graph, rel_type, source_label, target_label, limit)

    # Aggregation
    st.caption("Count relationships or nodes per group, sorted by count, and keep the top N.")
    c1, c2, c3 = st.columns([2, 1.3, 1.5])
    with c1:
        group_by = st.selectbox("Group and count", cfg["aggregation_modes"])
    node_grouping = group_by in cfg["aggregation_modes"][:2]
    with c2:
        label = st.selectbox("Node label filter (n)", [ANY] + labels, disabled=not node_grouping)
    with c3:
        top_n = st.slider("LIMIT (top N)", min_value=1, max_value=graph.number_of_nodes(), value=5)
    return query_aggregation(graph, group_by, label if node_grouping else ANY, top_n)


def reset_cypher_editor():
    st.session_state["cypher_editor"] = st.session_state.get("cypher_generated", "")


def render_cypher_editor(graph, builder_result):
    """Editable Cypher box: starts as the builder's query; an edited query runs through the Cypher subset.
    Returns the result to display, or None when the edited query cannot be run."""
    generated = builder_result["cypher"]
    # A new builder query (or returning to this section) refills the editor
    if st.session_state.get("cypher_generated") != generated or "cypher_editor" not in st.session_state:
        st.session_state["cypher_generated"] = generated
        st.session_state["cypher_editor"] = generated

    st.markdown("**Query Pattern** (read / write: edit it, then press Ctrl+Enter to run)")
    text = st.text_area("Query pattern", key="cypher_editor", label_visibility="collapsed",
                        height=max(110, 26 * (generated.count("\n") + 2)))
    edited = " ".join(text.split()) != " ".join(generated.split())

    col_reset, col_help = st.columns([1, 3])
    with col_reset:
        st.button("Reset to Query Builder", on_click=reset_cypher_editor, disabled=not edited, width="stretch")
    with col_help:
        st.caption("Supported: MATCH (patterns, -[*1..3]- paths), WHERE (AND / OR / NOT, =, <>, <, >, CONTAINS, "
                   "STARTS WITH, IN, IS NULL), RETURN (DISTINCT, AS, count, type, labels, length, properties), "
                   "ORDER BY, SKIP, LIMIT. The graph is read-only: CREATE, SET and DELETE are blocked.")

    if not edited:
        return builder_result
    try:
        return query_custom(graph, text)
    except CypherError as err:
        st.error(f"Could not run this query: {err}")
        return None


def render_simulation_section():
    """Renders Section 2: Graph Explorer, Query Builder, Results, and Trial Logger."""
    st.header("Interactive Simulation: Querying a Knowledge Graph")
    st.info("Explore the sample graph, build a query pattern with the guided Query Builder (or edit it by hand), "
            "and log each query as a trial. The graph lives in memory (networkx).")

    graph = build_knowledge_graph()
    pos = compute_layout(SIMULATION_CONFIG["layout_seed"],
                         tuple(sorted(SIMULATION_CONFIG["node_positions"].items())))

    render_graph_explorer(graph, pos)
    st.divider()
    result = render_query_builder(graph)

    result = render_cypher_editor(graph, result)

    if result is None:
        st.info("Fix the query above, or click 'Reset to Query Builder' to get the generated query back.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Rows Returned", len(result["table"]))
        with m2:
            st.metric("Matched Nodes", len(result["nodes"]))
        with m3:
            st.metric("Matched Relationships", len(result["edges"]))
        with m4:
            st.metric("Query Mode", result["mode"])
        st.write(f"**Result summary:** {result['summary']}")

        col_table, col_graph = st.columns([2, 3])
        with col_table:
            st.markdown("**Result Table**")
            if result["table"].empty:
                st.warning("The query returned 0 rows. Try a different label, relationship type, or direction.")
            else:
                st.dataframe(result["table"], width="stretch", hide_index=True)
        with col_graph:
            st.markdown("**Highlighted Subgraph** (matched items in red; start node enlarged)")
            st.plotly_chart(
                build_graph_figure(graph, pos, highlight_nodes=result["nodes"],
                                   highlight_edges=result["edges"], focus_nodes=result["focus"]),
                key="result_graph_chart"
            )

        if result["chart"] is not None and not result["chart"].empty:
            columns = list(result["table"].columns)
            st.plotly_chart(build_aggregation_chart(result["chart"], columns[-2], columns[-1]),
                            key="aggregation_chart")

    # Data Logger
    st.divider()
    st.subheader("Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Capture the current query and its result counts into your session trial table:")
        if st.button("Record Current Trial", type="primary", width="stretch", disabled=result is None):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Query Mode": result["mode"],
                "Scope": result["scope"],
                "Rows": len(result["table"]),
                "Nodes Hit": len(result["nodes"]),
                "Edges Hit": len(result["edges"]),
                "Cypher Query": " ".join(result["cypher"].split()),
                "Summary": result["summary"],
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully saved!")

        if st.button("Clear Logged Trials", width="stretch"):
            st.session_state["trials"] = []
            st.toast("Trial log cleared.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, width="stretch", hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Trials as CSV",
                data=csv_data,
                file_name="cypher_query_trials.csv",
                mime="text/csv",
                width="stretch"
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to begin collecting experimental data.")


def render_quiz_section():
    """Renders Section 3: Assessment Quiz with Self-Grading and Feedback."""
    st.header("Concept Assessment Quiz")
    st.write("Answer the conceptual questions below to evaluate your understanding of knowledge graphs and Cypher.")

    with st.form("lab_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.subheader(f"Question {q['id']}")
            st.write(q["question"])
            selected = st.radio(
                label=f"Options for Question {q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_radio_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)

        submitted = st.form_submit_button("Submit Quiz for Grading", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results and Feedback")
        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                         f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                         f"**Reasoning:** _{q['explanation']}_")

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.0f}%)")

    elif st.session_state.get("quiz_submitted", False):
        st.success(f"Quiz already submitted. Current score: **{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}**")


def render_report_section():
    """Renders Section 4: Dynamic Lab Report Generator with Guaranteed PDF Export."""
    st.header("Report Generation")
    st.write("Compile your student details, recorded query trials, and quiz evaluation into an official PDF report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Student Roll / ID", value=st.session_state["student_info"].get("id", "EXP-001"))
    with col3:
        lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Discussion & Observations")
    student_notes = st.text_area(
        "Enter your interpretation of the query results, observations, and conclusions:",
        value=st.session_state.get("student_notes") or DEFAULT_NOTES,
        height=140
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    st.write(f"**Experiment:** {EXPERIMENT_CONFIG['experiment_no']}: {EXPERIMENT_CONFIG['title']}")
    st.write(f"**Student:** {student_name} | **ID:** {student_id} | **Date:** {lab_date}")
    st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, width="stretch")
    else:
        st.info("Note: You have not recorded any query trials in the Simulation tab yet. Your report will indicate 0 trials.")

    # Generate PDF bytes and write file to disk
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes
    )

    # Save next to app.py so Streamlit static serving (./static) finds it regardless of working directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(app_dir, "static"), exist_ok=True)
    with open(os.path.join(app_dir, "static", "lab_report.pdf"), "wb") as f:
        f.write(pdf_bytes)
    with open(os.path.join(app_dir, "lab_report.pdf"), "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Lab Report (.pdf)")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        # Direct static link ending in .pdf (requires server.enableStaticServing, see .streamlit/config.toml)
        st.link_button(
            "Open / Download PDF Document",
            url="/app/static/lab_report.pdf",
            type="primary",
            width="stretch"
        )

    with col_btn2:
        # Standard Streamlit download button
        st.download_button(
            label="Download lab_report.pdf",
            data=pdf_bytes,
            file_name="lab_report.pdf",
            mime="application/pdf",
            key="stream_pdf_btn",
            width="stretch"
        )


# ======================================================================================
# 6. MAIN ENTRYPOINT & NAVIGATION
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP-001",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""


def main():
    st.set_page_config(
        page_title="KGIRS Lab - Cypher Queries",
        page_icon=None,
        layout="wide"
    )

    init_session_state()

    # Native Streamlit Title (No custom CSS)
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(f"{EXPERIMENT_CONFIG['experiment_no']} | {EXPERIMENT_CONFIG['course']}")

    # Navigation Sidebar
    section = st.sidebar.radio(
        "Lab Navigator",
        options=["Theory", "Simulation", "Quiz", "Report Generation"]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Progress Tracker")
    quiz_status = "Done" if st.session_state.get("quiz_submitted", False) else "Pending"
    st.sidebar.write(f"- **Quiz Status:** {quiz_status}")
    if st.session_state.get("quiz_submitted", False):
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    # Section Dispatcher
    if section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
