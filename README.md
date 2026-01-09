# Mobius-DocumentExtraction

Extract documents (.pdf, .docx, .epub, images, .ppt) to Markdown and chunks.

## Installation

```bash
pip install -r requirements-exact.txt
```

Set up environment variables:
```bash
# .env file
OPENAI_API_KEY=your_api_key_here
```

## Quick Start

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

converter = PdfConverter(
    artifact_dict=create_model_dict(),
    config={"page_range": [0, 1, 2], "renderer": "chunks+pageMarkdown"}
)
result = converter.render_document(converter.build_document("file.pdf"))
```

## RAG Chat Agent

Interactive CLI for PDF Q&A with caching.

```bash
python rag_chat.py --pdf-dir ./PDFs
python rag_chat.py --pdf ./doc.pdf --page-range 0-9
python rag_chat.py --clear-cache
```

**Commands:** `/list`, `/add <path>`, `/reload <path>`, `/cache`, `/cache clear`, `/quit`

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `page_range` | all | Pages to process |
| `renderer` | `json` | `markdown`, `json`, `pageMarkdown`, `chunks`, `html` |
| `disable_ocr` | `True` | Disable OCR |
| `use_llm` | `False` | Use LLM for enhanced parsing |
