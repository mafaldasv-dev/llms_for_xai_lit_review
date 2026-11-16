# LLMs for XAI — Systematic Literature Review

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Companion repository for the systematic literature review **"Application of Large Language Models in Explainable AI"**, covering the search, deduplication, venue-quality ranking and filtering, and quantitative/qualitative analysis pipeline behind the paper.

## Quick start

The notebook has already been run end to end, so its outputs are saved in place — you don't need to re-run it to inspect the results. If you do want to reproduce them, open `notebooks/script_for_literature_review.ipynb` and run top to bottom, after installing uv and the dependencies; note that the fuzzy-matching steps (deduplication and ranking) can take a while to complete.

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Repository structure
```
.
├── litreview/              # Reusable package: data loaders, venue-ranking, deduplication, analysis/plots
├── notebooks/
│   ├── script_for_literature_review.ipynb   # Main pipeline: dedup → ranking/filtering → quantitative & qualitative analysis
│   └── data/                                # Search results, ranking tables, processed outputs (see below)
├── requirements.txt
└── LICENSE
```

## Data (`notebooks/data/`)

| Path | Contents |
|---|---|
| `all_initial_results_combined.csv` | Raw merged output of all four database exports before deduplication |
| `databases_search_results/` | The four raw .bib/.csv database exports (ACM, Scopus, WoS, IEEE) |
| `dois_for_zotero.txt` | Plain-text DOI list, manually exported for importing into Zotero |
| `excluded_from_screening.csv` | Papers dropped during ranking filtering stage |
| `final_results_for_screening.csv` | Final list of papers after deduplication and filtering. These were the papers that moved to the screening phase, conducted with ASReview Lab software |
| `ranks/` | Folder containing venue-ranking reference tables used for filtering |
| `processed/asreview_relevant_final_results_for_qa_phase.csv` | The 152 ASReview-relevant studies after title/abstract screening. These were the papers that moved to the next phase, QA |
| `processed/data_form_extraction_and_qa_analysis_of_selected_papers.xlsx` | A copy of the excel that contains QA scores and data form extraction. The link for the excel is [here](https://docs.google.com/spreadsheets/d/1R-sxbGApWWWG6wV3rcTMz46Ww01rRbh1/edit?usp=sharing&ouid=100001465766847447164&rtpof=true&sd=true). |
| `processed/included_studies.bib` | BibTeX of the 50 included studies, used for the keyword co-occurrence network, the log-linear year counts, and the HHI calculation |
| `processed/included_studies_keyword_cooccurrence_vosviewer.json` | VOSviewer-format keyword co-occurrence network (Figure 6 in the paper) |

## The paper

**Application of Large Language Models in Explainable AI: A Systematic Literature Review**
Mafalda Sá-Velho, Mauro Castelli, João Fonseca

> The growing complexity of machine learning and deep learning models, particularly black-box architectures, has created a pressing demand for Explainable Artificial Intelligence (XAI) methods that make these models more transparent and interpretable. The rapid rise of Large Language Models (LLMs) has opened up new possibilities for generating intuitive and accessible explanations of AI systems. This paper presents a systematic literature review of how LLMs are being applied to improve the explainability of black-box AI systems. Following a PRISMA-based approach, a search was conducted on 18 January 2026 across Scopus, ACM Digital Library, IEEE Xplore and Web of Science, covering studies published between January 2022 and January 2026. Fifty peer-reviewed studies were included in the final review. We examined the roles that LLMs play in XAI, how they are integrated with XAI methods, how LLM-based XAI is evaluated, and the main challenges and research gaps. The review makes three contributions: a taxonomy of six functional roles of LLMs in XAI; a two-axis characterisation that separates the XAI method from the LLM integration technique; and a framework separating five dimensions of evaluation practice.

**Main conclusions**
- LLMs play six distinct functional roles in XAI (explanation generator, enhancer, conversational interface, self-explaining predictor, evaluator, reasoner); explanation generation is by far the most common (42/50 studies).
- LLMs are used both alongside established XAI methods like SHAP/LIME (33 hybrid studies) and as standalone explainers (17 LLM-only studies); no included study compares the two approaches under equivalent conditions.
- The evidence supports LLMs as a communication layer for existing XAI methods more clearly than as a replacement for them.
- Evaluation practice is fragmented: predictive-performance and linguistic-quality metrics (e.g. BLEU/ROUGE) receive far more attention than direct assessment of faithfulness and behavioural impact.
- A central, largely unresolved tension: using an opaque LLM to explain another opaque model risks producing fluent, plausible-sounding narratives that do not faithfully reflect the model's actual reasoning.
- Recurring challenges include hallucination, output variability across API versions, computational cost, domain adaptation, privacy, and fairness — only 9/50 studies assess behavioural impact through task-based measures.
- Future work should state each LLM's role explicitly, benchmark against conventional XAI baselines, and evaluate faithfulness separately from fluency and user perception.


## License
See the [LICENSE](LICENSE) file for details.
