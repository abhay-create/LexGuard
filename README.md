# LeXGuard: Legal Contract Compliance Verification System

## 1. Project Goal

LeXGuard is a clause-level legal compliance verification system that checks each clause in a contract against a versioned regulatory compliance knowledge base. The system should:

- ingest legal and policy documents in PDF, DOCX, and HTML formats,
- segment them into structured clauses,
- store the compliance corpus in an appendable and editable knowledge base,
- retrieve the most relevant regulatory evidence for each contract clause,
- decide whether the clause is compliant or conflicting,
- produce evidence-grounded explanations with version metadata,
- support auditability, amendment tracking, and human review.

The design is optimized for a local development machine with a 16 GB VRAM GPU and standard CPU resources.

---

## 2. Problem Framing

### Inputs
- **Contract document**: a clause-containing legal document.
- **Compliance knowledge base**: policies, laws, regulations, internal standards, and amendments.

### Output for each contract clause
- **Decision label**:
  - Compliant
  - Conflicting
  - Partially compliant
  - Missing required clause
  - Needs human review
  - Not applicable
- **Evidence**:
  - relevant compliance clause(s)
  - source document identifier
  - version metadata
  - effective date range
- **Explanation**:
  - plain-language justification grounded only in retrieved evidence

### Core constraints
- no training dataset is available,
- no validation dataset is available,
- fine-tuning is not assumed,
- the system must remain feasible on 16 GB VRAM.

---

## 3. Architectural Principles

1. **Evidence first**: every decision must be backed by retrieved text and metadata.
2. **Version-aware knowledge**: legal content changes, so the knowledge base must preserve history.
3. **Hybrid retrieval**: lexical matching and semantic search should both be used.
4. **Zero-shot decisioning**: because no labeled data is available, the decision layer remains zero-shot.
5. **Grounded explanation**: the LLM explains the result, but never decides it.
6. **Human review fallback**: uncertain or low-confidence cases should not be forced into a hard label.
7. **Auditability**: every result should be traceable to a specific version of source text.

---

## 4. High-Level System Architecture

```text
                ┌──────────────────────────────┐
                │  Contract / Compliance Docs   │
                │  PDF / DOCX / HTML / TXT      │
                └──────────────┬───────────────┘
                               │
                               v
                ┌──────────────────────────────┐
                │  Ingestion & Parsing Layer    │
                │  Docling + Unstructured       │
                └──────────────┬───────────────┘
                               │
                               v
                ┌──────────────────────────────┐
                │  Clause Segmentation Layer    │
                │  headings / numbering / spans │
                └──────────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                v                             v
     ┌──────────────────────┐      ┌──────────────────────────┐
     │ Versioned RDF Graph   │      │ Searchable Text Indexes   │
     │ provenance, lineage   │      │ BM25 + dense vectors      │
     └──────────┬───────────┘      └───────────┬──────────────┘
                │                               │
                └──────────────┬────────────────┘
                               v
                ┌──────────────────────────────┐
                │  Retrieval Fusion Layer       │
                │  RRF + filtering              │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  Reranking Layer              │
                │  bge-reranker-v2-m3           │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  Decision Layer               │
                │  zero-shot NLI + rules        │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  Explanation Layer            │
                │  quantized local LLM          │
                └──────────────┬───────────────┘
                               v
                ┌──────────────────────────────┐
                │  Output / Audit Report        │
                └──────────────────────────────┘
```

---

## 5. Recommended Repository Layout

```text
lexguard/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── config/
│   ├── app.yaml
│   ├── logging.yaml
│   ├── retrieval.yaml
│   ├── rdf_schema.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   │   ├── contracts/
│   │   └── compliance_kb/
│   ├── processed/
│   │   ├── parsed_docs/
│   │   ├── clauses/
│   │   ├── embeddings/
│   │   └── graphs/
│   └── benchmarks/
│       ├── contractnli/
│       ├── cuad/
│       ├── legalbench/
│       └── synthetic/
├── src/
│   └── lexguard/
│       ├── __init__.py
│       ├── main.py
│       ├── settings.py
│       ├── logging_utils.py
│       ├── schemas/
│       │   ├── document.py
│       │   ├── clause.py
│       │   ├── evidence.py
│       │   ├── verdict.py
│       │   └── graph.py
│       ├── ingestion/
│       │   ├── docling_parser.py
│       │   ├── unstructured_parser.py
│       │   ├── clause_segmenter.py
│       │   ├── text_cleaner.py
│       │   └── provenance.py
│       ├── knowledge_base/
│       │   ├── rdf_store.py
│       │   ├── schema_manager.py
│       │   ├── versioning.py
│       │   ├── lineage.py
│       │   └── query.py
│       ├── indexing/
│       │   ├── bm25_index.py
│       │   ├── dense_index.py
│       │   ├── vector_utils.py
│       │   └── fusion.py
│       ├── retrieval/
│       │   ├── hybrid_retriever.py
│       │   ├── filters.py
│       │   └── candidate_builder.py
│       ├── reranking/
│       │   ├── reranker.py
│       │   └── candidate_sorter.py
│       ├── decision/
│       │   ├── nli_decision.py
│       │   ├── label_mapper.py
│       │   ├── confidence.py
│       │   └── human_review_rules.py
│       ├── explanation/
│       │   ├── prompt_builder.py
│       │   ├── generator.py
│       │   └── grounding.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── benchmark_runner.py
│       │   ├── regression_tests.py
│       │   └── report_builder.py
│       ├── pipeline/
│       │   ├── process_contract.py
│       │   ├── process_kb.py
│       │   └── batch_runner.py
│       ├── api/
│       │   ├── app.py
│       │   ├── routes.py
│       │   └── models.py
│       └── utils/
│           ├── tokenizer.py
│           ├── io.py
│           ├── dates.py
│           ├── ids.py
│           └── batching.py
├── scripts/
│   ├── build_kb.py
│   ├── ingest_contract.py
│   ├── build_indexes.py
│   ├── run_benchmark.py
│   ├── export_report.py
│   └── synthetic_data.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_segmentation.py
│   ├── test_versioning.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_decision.py
│   ├── test_explanation.py
│   └── test_pipeline.py
└── docs/
    ├── architecture.md
    ├── rdf_schema.md
    ├── evaluation.md
    ├── api.md
    └── demo.md
```

---

## 6. Module-by-Module Implementation Detail

## 6.1 Ingestion Layer

### Purpose
Parse source documents into structured text with page references, section hierarchy, and provenance.

### Responsibilities
- read PDF, DOCX, HTML, and plain text,
- preserve document structure,
- extract metadata,
- detect failed parses,
- emit structured intermediate JSON.

### Files
- `docling_parser.py`
- `unstructured_parser.py`
- `text_cleaner.py`
- `provenance.py`

### Implementation steps
1. Use Docling as the primary parser.
2. If a document fails or produces poor structure, use Unstructured as fallback.
3. Normalize all parser outputs into a common schema.
4. Attach source metadata:
   - `document_id`
   - `source_path`
   - `page_number`
   - `block_id`
   - `section_heading`
   - `parse_engine`
5. Store the raw parser output for debugging.

### Output schema
```json
{
  "document_id": "DOC_001",
  "source_path": "data/raw/contracts/contract_a.pdf",
  "parse_engine": "docling",
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "block_id": "B1",
          "text": "...",
          "bbox": [0.1, 0.2, 0.9, 0.3],
          "block_type": "paragraph"
        }
      ]
    }
  ]
}
```

---

## 6.2 Clause Segmentation Layer

### Purpose
Convert parsed document blocks into clause-level units.

### Responsibilities
- detect headings,
- detect numbered items,
- detect subclauses,
- merge wrapped lines,
- preserve clause hierarchy,
- assign stable clause IDs.

### Files
- `clause_segmenter.py`
- `tokenizer.py`
- `ids.py`

### Segmentation rules
- split on clause numbering patterns:
  - `1.`
  - `1.1`
  - `4(a)`
  - `(i)`
- preserve subclause nesting,
- merge lines that belong to the same logical clause,
- keep tables as separate clause candidates if they encode obligations,
- keep defined term sections and exception sections linked to parent clauses.

### Clause output schema
```json
{
  "clause_id": "CONTRACT_A_C12",
  "document_id": "CONTRACT_A",
  "section_path": ["4", "4.2", "4.2(a)"],
  "text": "The vendor shall...",
  "page_start": 5,
  "page_end": 5,
  "parent_clause_id": "CONTRACT_A_C11",
  "clause_type": "obligation",
  "source_spans": [
    {
      "page": 5,
      "start_char": 120,
      "end_char": 487
    }
  ]
}
```

### Important note
Do not assume the parser gives clauses directly. Clause segmentation must be built on top of parsed blocks.

---

## 6.3 Knowledge Base Layer

### Purpose
Store compliance content as a versioned, editable, appendable knowledge base with explicit lineage.

### Recommended design
Use an RDF graph for:
- compliance clause records,
- source provenance,
- version links,
- effective dates,
- supersession relationships,
- cross references between provisions.

### Files
- `rdf_store.py`
- `schema_manager.py`
- `versioning.py`
- `lineage.py`
- `query.py`

### Core RDF entities
- `Provision`
- `DocumentVersion`
- `SourceDocument`
- `Entity`
- `Jurisdiction`
- `DefinedTerm`

### Core RDF properties
- `hasText`
- `versionId`
- `effectiveFrom`
- `effectiveTo`
- `status`
- `supersedes`
- `supersededBy`
- `derivedFrom`
- `mentionsEntity`
- `referencesProvision`
- `hasJurisdiction`

### Versioning rules
- never overwrite previous versions,
- create a new version node for every amendment,
- mark the previous version as superseded,
- keep effective date windows,
- query only active versions at inference time,
- preserve lineage for audit and regression testing.

### Example triple structure
```turtle
lex:prov_001_v1 a lex:Provision ;
    lex:provisionId "PROV_001" ;
    lex:versionId "PROV_001_V1" ;
    lex:hasText "..." ;
    lex:effectiveFrom "2024-01-01"^^xsd:date ;
    lex:effectiveTo "2024-12-31"^^xsd:date ;
    lex:status "superseded" ;
    lex:supersededBy lex:prov_001_v2 .
```

### Query behavior
At inference time, return only provisions where:
- `status = active`, or
- `effectiveFrom <= document_date <= effectiveTo`

### Why RDF is useful here
The graph layer gives you production-style features:
- lineage,
- amendment tracking,
- cross-linking,
- version-aware queries,
- explainable evidence trails.

---

## 6.4 Indexing Layer

### Purpose
Build searchable indexes for hybrid retrieval.

### Files
- `bm25_index.py`
- `dense_index.py`
- `vector_utils.py`
- `fusion.py`

### BM25 index
Use BM25 for:
- exact legal phrase matching,
- section references,
- defined terms,
- obligation keywords,
- exception phrasing.

### BM25 preprocessing rules
Replace naive token splitting with a legal-aware tokenizer that preserves:
- citation forms,
- section numbers,
- defined terms,
- punctuation in legal references,
- clause boundary markers.

### Dense index
Use BGE-M3 embeddings for semantic retrieval.

### Dense vector handling
- encode provisions in batches,
- normalize embeddings before indexing,
- use cosine-equivalent inner product search,
- persist embeddings to disk.

### Fusion
Use Reciprocal Rank Fusion (RRF) to combine BM25 and dense rankings.

### Example fusion flow
1. Retrieve top 100 BM25 matches.
2. Retrieve top 100 dense matches.
3. Merge with RRF.
4. Keep top 20 or top 50 candidates.
5. Pass to reranker.

---

## 6.5 Retrieval Layer

### Purpose
Return the most relevant compliance provisions for each contract clause.

### Files
- `hybrid_retriever.py`
- `filters.py`
- `candidate_builder.py`

### Retrieval stages
1. build query from clause text,
2. apply metadata filters,
3. search BM25,
4. search dense index,
5. fuse results,
6. rerank fused candidates.

### Metadata filters
- jurisdiction,
- active version,
- effective date window,
- document type,
- policy category,
- source trust level.

### Output schema
```json
{
  "clause_id": "CONTRACT_A_C12",
  "candidate_ids": ["PROV_88", "PROV_41"],
  "retrieval_scores": [0.82, 0.77]
}
```

---

## 6.6 Reranking Layer

### Purpose
Improve retrieval precision by rescoring candidate clause pairs.

### Files
- `reranker.py`
- `candidate_sorter.py`

### Reranker choice
Use `bge-reranker-v2-m3`.

### Behavior
- input: contract clause + candidate compliance clause,
- output: relevance score,
- sort candidates by score,
- keep top evidence snippets.

### Important implementation detail
Reranking must operate on one clause at a time or on correctly batched clause-candidate pairs with stable mapping metadata. Do not mix scores across clauses.

### Candidate record schema
```json
{
  "clause_id": "CONTRACT_A_C12",
  "candidate_id": "PROV_41",
  "dense_score": 0.66,
  "bm25_score": 12.5,
  "rrf_score": 0.021,
  "rerank_score": 0.91
}
```

---

## 6.7 Decision Layer

### Purpose
Map retrieved evidence to a compliance verdict without fine-tuning.

### Files
- `nli_decision.py`
- `label_mapper.py`
- `confidence.py`
- `human_review_rules.py`

### Model strategy
Use a zero-shot NLI model as the baseline judgment layer, but do not rely on it blindly.

### Decision labels
- Compliant
- Conflicting
- Partially compliant
- Missing required clause
- Needs human review
- Not applicable

### Decision logic
A practical rule-based mapping should combine:
- reranker score,
- NLI entailment score,
- NLI contradiction score,
- evidence coverage,
- version match,
- jurisdiction match,
- confidence threshold.

### Example decision rules
- **Compliant**: high entailment, no contradiction, required evidence found.
- **Conflicting**: high contradiction against active provision.
- **Partially compliant**: some support exists, but required conditions are missing.
- **Missing required clause**: no matching obligation or no mandatory coverage found.
- **Not applicable**: clause is outside the KB scope or out of jurisdiction.
- **Needs human review**: scores are ambiguous or below threshold.

### Confidence handling
If confidence is below threshold or evidence is sparse, do not force a hard label. Escalate to human review.

---

## 6.8 Explanation Layer

### Purpose
Generate a readable summary that explains the decision using only retrieved evidence.

### Files
- `prompt_builder.py`
- `generator.py`
- `grounding.py`

### Model strategy
Use a quantized local LLM such as Llama 3.1 8B Instruct.

### Hard constraints
- explain only the chosen evidence,
- do not invent legal reasoning,
- do not add unsupported claims,
- if evidence is weak, explicitly say the case needs human review.

### Explanation input bundle
- clause text,
- final label,
- confidence score,
- top evidence text,
- version metadata,
- line-by-line supporting citations.

### Example prompt structure
```text
You are a legal compliance assistant.

Contract clause:
...

Decision:
Conflicting

Evidence:
- Provision text: ...
- Version: v3
- Effective from: 2024-01-01
- Effective to: 2024-12-31

Write a short, grounded explanation for a compliance officer.
Only use the evidence above.
```

---

## 6.9 Evaluation Layer

### Purpose
Measure quality, robustness, and regression behavior.

### Files
- `metrics.py`
- `benchmark_runner.py`
- `regression_tests.py`
- `report_builder.py`

### Metrics to track
- retrieval recall@k
- mean reciprocal rank
- reranking NDCG
- decision accuracy
- macro F1 by class
- contradiction detection accuracy
- evidence precision
- evidence recall
- version regression pass rate
- amendment consistency score
- false negative rate for critical clauses
- human review rate

### Evaluation strategy
Use multiple benchmarks because no company dataset is available.

### Recommended public datasets
- ContractNLI for entailment and evidence
- CUAD for clause-level contract tasks
- LEDGAR for clause classification
- LegalBench for broad legal reasoning
- LRAGE for legal-domain RAG-style evaluation

### Synthetic evaluation set
Create a small local benchmark by:
- taking public contract clauses,
- editing them synthetically,
- inserting version changes into the KB,
- testing whether the system detects conflicts, omissions, and amendments.

---

## 6.10 Pipeline Layer

### Purpose
Orchestrate end-to-end processing of a contract against the compliance KB.

### Files
- `process_contract.py`
- `process_kb.py`
- `batch_runner.py`

### Pipeline steps
1. parse input contract,
2. segment into clauses,
3. retrieve matching KB provisions,
4. rerank candidates,
5. run decision layer,
6. generate grounded explanation,
7. store results,
8. export audit report.

### Batch processing rules
- batch by clause, not by arbitrary text chunks,
- keep clause-to-evidence mapping intact,
- store per-clause intermediate artifacts,
- batch retrieval and reranking where safe,
- do not merge scores from different clauses.

### Output artifact
Each processed clause should produce a result record with:
- clause text
- final label
- confidence
- top evidence
- source version metadata
- explanation
- review flag

---

## 6.11 API Layer

### Purpose
Expose the system through a simple service interface.

### Files
- `app.py`
- `routes.py`
- `models.py`

### Suggested endpoints
- `POST /ingest/kb`
- `POST /ingest/contract`
- `POST /process/contract`
- `GET /results/{job_id}`
- `GET /kb/versions/{provision_id}`
- `GET /audit/{job_id}`

### API behavior
- accept document uploads,
- return job identifiers,
- provide clause-wise results,
- expose evidence and version traces,
- support asynchronous processing if needed.

---

## 6.12 Utility Layer

### Purpose
Provide reusable helper functions.

### Files
- `tokenizer.py`
- `io.py`
- `dates.py`
- `ids.py`
- `batching.py`

### Common utilities
- legal tokenizer,
- version/date normalization,
- stable ID generation,
- batching helpers,
- safe file I/O,
- config loading.

---

## 7. Data Model Definitions

## 7.1 Document model
```json
{
  "document_id": "string",
  "doc_type": "contract | policy | regulation",
  "source_path": "string",
  "version": "string",
  "effective_from": "YYYY-MM-DD",
  "effective_to": "YYYY-MM-DD",
  "jurisdiction": "string",
  "parse_engine": "string"
}
```

## 7.2 Clause model
```json
{
  "clause_id": "string",
  "document_id": "string",
  "section_path": ["string"],
  "text": "string",
  "page_start": 1,
  "page_end": 1,
  "clause_type": "string",
  "parent_clause_id": "string|null",
  "source_spans": []
}
```

## 7.3 Provision model
```json
{
  "provision_id": "string",
  "version_id": "string",
  "document_id": "string",
  "text": "string",
  "effective_from": "YYYY-MM-DD",
  "effective_to": "YYYY-MM-DD",
  "status": "active | superseded | draft",
  "jurisdiction": "string",
  "mentions": ["string"]
}
```

## 7.4 Result model
```json
{
  "clause_id": "string",
  "final_label": "Compliant | Conflicting | Partially compliant | Missing required clause | Needs human review | Not applicable",
  "confidence": 0.0,
  "evidence": [
    {
      "provision_id": "string",
      "version_id": "string",
      "text": "string",
      "score": 0.0
    }
  ],
  "explanation": "string",
  "review_required": true
}
```

---

## 8. Open-Data Simulation Plan

Since no company data is available, the system should be validated using public legal datasets.

### Recommended datasets
- **CUAD**: clause extraction and legal clause coverage
- **ContractNLI**: entailment and evidence span evaluation
- **LEDGAR**: legal clause classification
- **LegalBench**: general legal reasoning tasks
- **LRAGE**: legal-domain retrieval augmented generation evaluation

### Simulation strategy
1. Build a compliance KB from public legal and policy-like text.
2. Use public contract datasets as the contract side.
3. Treat clauses from the contract as queries.
4. Retrieve matching KB provisions.
5. Apply versioned compliance logic.
6. Compare predictions against derived labels or synthetic labels.

### Synthetic amendments
Create artificial version changes such as:
- updated obligations,
- new exceptions,
- removed clauses,
- effective date changes,
- jurisdiction-specific overrides.

This helps test versioning and regression handling.

---

## 9. Resource and Deployment Plan

### Target environment
- 16 GB GPU
- standard CPU
- local development and demo execution

### Memory strategy
- use quantized LLM for explanation only,
- batch retrieval and reranking,
- store embeddings on disk,
- keep graph operations lightweight,
- avoid unnecessary full-document loading in GPU memory.

### Suggested model roles
- **BGE-M3**: embeddings / retrieval
- **bge-reranker-v2-m3**: reranking
- **zero-shot NLI model**: decision baseline
- **Llama 3.1 8B Instruct (quantized)**: explanation

### Practical note
The system should be designed so that the LLM is not active during every sub-step unless explanation is requested.

---

## 10. Testing Strategy

### Unit tests
- parser output consistency
- clause segmentation correctness
- versioning links
- BM25 tokenization
- dense embedding normalization
- RRF merging correctness
- reranker score alignment
- decision label mapping
- explanation grounding

### Integration tests
- full contract-to-result pipeline
- KB update followed by re-query
- amendment supersession behavior
- human review fallback path

### Regression tests
- previous results should stay stable after KB updates unless a version change should alter them
- active version queries should exclude superseded provisions

---

## 11. Implementation Order

### Phase 1
- set up repository structure,
- define schemas,
- build parser and clause segmentation,
- implement RDF versioning.

### Phase 2
- build BM25 and dense indexes,
- normalize embeddings,
- implement retrieval fusion.

### Phase 3
- add reranking,
- add zero-shot decision logic,
- create confidence thresholds.

### Phase 4
- add explanation generation,
- ground explanations in evidence only.

### Phase 5
- implement evaluation scripts,
- run public benchmarks,
- build synthetic amendments.

### Phase 6
- add API endpoints,
- export reports,
- prepare demo workflow.

---

## 12. Demo Workflow for Interview

A strong demo should show:
1. upload a contract,
2. parse and segment clauses,
3. query the compliance KB,
4. show retrieved evidence,
5. show version-aware filtering,
6. output the decision label,
7. show the plain-language explanation,
8. show why some clauses are sent to human review,
9. show that amendments change the result when version metadata changes.

---

## 13. Key Design Choices to Emphasize

- RDF is used for **auditability and lineage**, not as a magical replacement for indexing.
- Hybrid retrieval is used because legal text depends on both exact wording and semantic similarity.
- Zero-shot NLI is used because there is no training data.
- The LLM is used only for grounded explanation.
- Human review is a first-class outcome, not a failure state.
- The whole system is built to run on limited GPU resources.

---

## 14. Final Notes

This project is best presented as a **version-aware, evidence-grounded legal compliance verification system** rather than a generic legal chatbot. The strongest interview story is that it combines:
- document intelligence,
- retrieval engineering,
- graph-based provenance,
- zero-shot legal reasoning,
- and audit-ready explanations.

That combination makes the project look production-relevant while staying realistic for a solo build on limited hardware.
