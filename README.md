# Experiment 10: Query Knowledge Graphs using Cypher

**Course:** Knowledge Graphs and Information Retrieval Systems (KGIRS) - Virtual Lab

## About the experiment

This virtual lab teaches how to query a knowledge graph with Cypher, the declarative, pattern-based
query language used by Neo4j. Students explore a small movie knowledge graph with 16 nodes
(people, movies, studios, cities) and 25 typed, directed relationships (`ACTED_IN`, `DIRECTED`,
`PRODUCED`, `DISTRIBUTED`, `FOUNDED`, `LOCATED_IN`, `BORN_IN`). A guided Query Builder turns their
choices into an equivalent Cypher query for five query patterns:
- node lookup with property filters
- one-hop relationship traversal with a chosen direction
- variable-length multi-hop traversal (`[*1..3]`)
- filtered pattern matching
- aggregation with `count()`, `ORDER BY` and `LIMIT`

Each query runs against the graph and returns a result table and a highlighted subgraph.
The generated Cypher can also be edited by hand. A small built-in interpreter runs the edited query
(read-only `MATCH` / `WHERE` / `RETURN` / `ORDER BY` / `SKIP` / `LIMIT`, including `count()` and
variable-length paths), and write clauses such as `CREATE`, `SET` or `DELETE` are blocked.
Students log their queries as trials, take a 10-question concept quiz, and download a PDF lab
report. As the course rule for graph experiments requires, the app uses no Neo4j database. The
graph is stored as plain Python data and loaded into an in-memory `networkx` graph, and the app
runs the same logic as each Cypher query directly on that graph.

## How to run

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Use the sidebar to move between the four sections:
**Theory -> Simulation -> Quiz -> Report Generation**.

## Files

| File | Purpose |
|---|---|
| `app.py` | The complete single-file Streamlit app (theory, graph data, query engine, plots, quiz, PDF report) |
| `requirements.txt` | Python dependencies: streamlit, pandas, numpy, plotly, networkx, fpdf2 |
| `.streamlit/config.toml` | Turns on static file serving so the "Open / Download PDF Document" link works |

When a report is generated, the app writes `lab_report.pdf` (and a copy in `static/`) next to `app.py`.
