# OVH Workbooks

A comprehensive collection of tutorials and workbooks for OVHcloud services.

## Quick Links

- **Documentation Site**: [https://cougz.github.io/ovhcloud-workbooks/](https://cougz.github.io/ovhcloud-workbooks/)
- **Source Repository**: [https://github.com/cougz/ovhcloud-workbooks](https://github.com/cougz/ovhcloud-workbooks)

## Available Workbooks

### Public Cloud

#### AI Endpoints
- **RAG Tutorial** - Build production-ready RAG systems using OVHcloud AI Endpoints

## Repository Structure

```
ovh-workbooks/
├── README.md                           # Main repo overview
├── mkdocs.yml                         # Root MkDocs configuration
├── docs/
│   ├── index.md                       # Main landing page
│   └── public-cloud/
│       └── ai-endpoints/
│           └── rag-tutorial/
│               ├── index.md           # RAG tutorial landing
│               ├── setup-guide.md     # Your step-by-step guide
│               └── scripts/           # Code snippets for docs
├── public-cloud/
│   └── ai-endpoints/
│       └── rag-tutorial/
│           ├── README.md              # Tutorial-specific readme
│           ├── scripts/               # Downloadable Python files
│           │   ├── test_ovh_connection.py
│           │   ├── test_rag_ovh.py
│           │   ├── performance_test.py
│           │   ├── model_comparison.py
│           │   ├── sensitivity_experiment.py
│           │   ├── my_rag_app.py
│           │   └── optimization_test.py
│           └── requirements.txt       # Dependencies
└── other-workbooks/                   # Future workbooks
```

## Development

This repository uses MkDocs with the Material theme to generate documentation. To run locally:

```bash
pip install mkdocs-material pymdown-extensions
mkdocs serve
```

## Deployment

The documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
