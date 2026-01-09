"""
RAG Chat Agent - Local Interface for PDF Document Q&A

This script provides a local command-line interface for conversing with
a RAG (Retrieval Augmented Generation) agent that can answer questions
about PDF documents.

Features:
- PDF caching: Processed PDFs are cached locally for faster subsequent loads
- Interactive chat interface with tool-augmented responses
- Support for multiple PDFs with RAG-based retrieval

Usage:
    python rag_chat.py [--pdf-dir <path>] [--page-range <start-end>]

Example:
    python rag_chat.py --pdf-dir ./PDFs --page-range 0-9
    python rag_chat.py --pdf-dir ./PDFs --no-cache  # Disable caching
    python rag_chat.py --clear-cache  # Clear all cached data
"""

import os
import sys
import json
import re
import ast
import argparse
import hashlib
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Type, Optional, List
from datetime import datetime

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Load environment variables
load_dotenv()

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rag_chat"


class PDFCache:
    """Manages caching of processed PDF data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the PDF cache.

        Args:
            cache_dir: Directory to store cached data (default: ~/.cache/rag_chat)
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load the cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self):
        """Save the cache index to disk."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _get_cache_key(self, pdf_path: str, page_range: List[int]) -> str:
        """Generate a unique cache key for a PDF with specific page range."""
        pdf_path = str(Path(pdf_path).resolve())
        # Include file modification time to invalidate cache when PDF changes
        try:
            mtime = os.path.getmtime(pdf_path)
        except OSError:
            mtime = 0

        key_string = f"{pdf_path}:{sorted(page_range)}:{mtime}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the path for a cached item."""
        return self.cache_dir / f"{cache_key}"

    def get(self, pdf_path: str, page_range: List[int]) -> Optional[dict]:
        """
        Retrieve cached PDF data if available.

        Args:
            pdf_path: Path to the PDF file
            page_range: List of page numbers that were processed

        Returns:
            Cached data dict or None if not cached
        """
        cache_key = self._get_cache_key(pdf_path, page_range)
        cache_path = self._get_cache_path(cache_key)

        if cache_key not in self.index:
            return None

        # Load doc_render and summary from JSON
        json_path = cache_path.with_suffix('.json')
        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load vectorstore from FAISS index if it exists
            faiss_path = cache_path.with_suffix('.faiss')
            vectorstore = None
            if faiss_path.exists():
                try:
                    from langchain_community.vectorstores import FAISS
                    from langchain_huggingface import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.load_local(
                        str(cache_path.with_suffix('')),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"  Warning: Could not load vectorstore from cache: {e}")

            return {
                'doc_render': data['doc_render'],
                'summary': data['summary'],
                'abstract': data.get('abstract'),
                'vectorstore': vectorstore,
                'pdf_id': data['pdf_id'],
                'cached_at': data.get('cached_at')
            }
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"  Warning: Cache corrupted for {pdf_path}: {e}")
            return None

    def set(self, pdf_path: str, page_range: List[int], data: dict):
        """
        Cache PDF data.

        Args:
            pdf_path: Path to the PDF file
            page_range: List of page numbers that were processed
            data: Dict containing doc_render, summary, vectorstore, etc.
        """
        cache_key = self._get_cache_key(pdf_path, page_range)
        cache_path = self._get_cache_path(cache_key)

        try:
            # Save doc_render and summary as JSON
            json_data = {
                'doc_render': data['doc_render'],
                'summary': data['summary'],
                'abstract': data.get('abstract'),
                'pdf_id': data['pdf_id'],
                'pdf_path': str(pdf_path),
                'page_range': page_range,
                'cached_at': datetime.now().isoformat()
            }
            with open(cache_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            # Save vectorstore using FAISS native format
            if data.get('vectorstore') is not None:
                data['vectorstore'].save_local(str(cache_path.with_suffix('')))

            # Update index
            self.index[cache_key] = {
                'pdf_path': str(pdf_path),
                'pdf_id': data['pdf_id'],
                'page_range': page_range,
                'cached_at': json_data['cached_at']
            }
            self._save_index()

        except Exception as e:
            print(f"  Warning: Could not cache {pdf_path}: {e}")

    def clear(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._save_index()
        print(f"Cache cleared: {self.cache_dir}")

    def list_cached(self) -> List[dict]:
        """List all cached PDFs."""
        return [
            {
                'pdf_id': info['pdf_id'],
                'pdf_path': info['pdf_path'],
                'page_range': info['page_range'],
                'cached_at': info['cached_at']
            }
            for info in self.index.values()
        ]


class PDFRAGAgent:
    """RAG Agent for PDF document Q&A with local conversation interface."""

    def __init__(
        self,
        pdf_paths: Optional[List[str]] = None,
        pdf_dir: Optional[str] = None,
        page_range: Optional[List[int]] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the RAG Agent.

        Args:
            pdf_paths: List of paths to PDF files
            pdf_dir: Directory containing PDF files (alternative to pdf_paths)
            page_range: List of page numbers to process (default: first 10 pages)
            model_name: OpenAI model name
            embedding_model: HuggingFace embedding model name
            use_cache: Whether to use caching for processed PDFs (default: True)
            cache_dir: Directory for cache storage (default: ~/.cache/rag_chat)
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.page_range = page_range or list(range(10))
        self.use_cache = use_cache
        self.cache = PDFCache(Path(cache_dir) if cache_dir else None) if use_cache else None

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Storage for PDF data
        self.pdfs = {}
        self.paths = []

        # Determine PDF paths
        if pdf_paths:
            self.paths = pdf_paths
        elif pdf_dir:
            pdf_dir_path = Path(pdf_dir)
            if pdf_dir_path.exists():
                self.paths = [str(p) for p in pdf_dir_path.glob("*.pdf")]
            else:
                print(f"Warning: PDF directory '{pdf_dir}' does not exist.")

        # Initialize tools and agent
        self._setup_tools()
        self._setup_graph()

    def load_pdfs(self):
        """Load and process PDF documents, using cache when available."""
        if not self.paths:
            print("No PDF files to load. Use add_pdf() to add documents.")
            return

        print(f"Loading {len(self.paths)} PDF document(s)...")

        # Track which PDFs need processing (not in cache)
        pdfs_to_process = []
        for path in self.paths:
            cached_data = None
            if self.use_cache and self.cache:
                cached_data = self.cache.get(path, self.page_range)

            if cached_data:
                print(f"  [CACHED] {Path(path).name}")
                self.pdfs[path] = {
                    'pdf_id': cached_data['pdf_id'],
                    'converter': None,  # Not needed for cached data
                    'doc_render': cached_data['doc_render'],
                    'summary': cached_data['summary'],
                    'abstract': cached_data.get('abstract'),
                    'vectorstore': cached_data['vectorstore']
                }
            else:
                pdfs_to_process.append(path)

        # Process PDFs that weren't cached
        if pdfs_to_process:
            print("Creating model dictionary (this may take a moment)...")
            artifact_dict = create_model_dict()

            for i, path in enumerate(pdfs_to_process, 1):
                print(f"  [{i}/{len(pdfs_to_process)}] Processing: {Path(path).name}")
                self._process_pdf(path, artifact_dict)

                # Cache the processed PDF
                if self.use_cache and self.cache and path in self.pdfs:
                    self.cache.set(path, self.page_range, self.pdfs[path])
                    print(f"    -> Cached for future use")

        print("PDF loading complete!")
        self._update_pdf_info()

    def _process_pdf(self, path: str, artifact_dict: dict):
        """Process a single PDF file."""
        try:
            converter = PdfConverter(
                artifact_dict=artifact_dict,
                config={
                    "page_range": self.page_range,
                    'disable_ocr': True,
                    'output_json': True,
                    "ignore_TOC": False,
                    'ignore_before_TOC': True,
                    "renderer": "chunks+pageMarkdown",
                    "disable_tqdm": True,
                },
                processor_list=[]
            )
            document = converter.build_document(path)
            doc_render = converter.render_document(document)
            doc_render.pop('metadata', None)

            # Generate summary
            summary = self._generate_summary(doc_render)

            # Build vector store
            chunks_text, metadata = self._extract_chunks(doc_render)
            vectorstore = FAISS.from_texts(
                chunks_text, self.embeddings, metadatas=metadata
            ) if chunks_text else None

            self.pdfs[path] = {
                'pdf_id': Path(path).name,
                'converter': converter,
                'doc_render': doc_render,
                'summary': summary,
                'vectorstore': vectorstore
            }
        except Exception as e:
            print(f"    Error processing {path}: {e}")

    def _generate_summary(self, doc_render: dict) -> str:
        """Generate a summary of the PDF document."""
        page_renders = doc_render.get('page_renders', [])
        if not page_renders:
            return "No content available for summarization."

        markdown = page_renders[0].get('markdown', '')
        if not markdown:
            return "No content available for summarization."

        prompt = HumanMessage(
            content=f"Following is the markdown of the first page of a document. "
                    f"Summarize the content in not more than 200 words:\n\n{markdown}"
        )
        response = self.llm.invoke([prompt])
        return response.content.strip()

    def _extract_chunks(self, doc_render: dict, chunk_size: int = 500, chunk_overlap: int = 50):
        """Extract text chunks from the document for RAG."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        plain_texts = []
        metadata = []

        for chunk in doc_render.get('chunks', {}).values():
            if chunk.get('block_type') != 'Text':
                continue

            html = chunk.get('html', '').strip()
            soup = BeautifulSoup(html, 'html.parser')
            plain_text = soup.get_text(separator=' ', strip=True)

            chunk_texts = text_splitter.split_text(plain_text)
            for text in chunk_texts:
                plain_texts.append(text)
                metadata.append({
                    "id": f"/page/{chunk.get('page')}/{chunk.get('block_type')}/{chunk.get('block_id')}",
                    "page": chunk.get('page')
                })

        return plain_texts, metadata

    def _find_chunk_by_id(self, chunk_id: str, doc_render: dict) -> str:
        """Find a chunk by its ID."""
        for current_id, chunk in doc_render.get('chunks', {}).items():
            if current_id == chunk_id:
                html = chunk.get('html', '').strip()
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator=' ', strip=True)
        raise ValueError(f"Chunk with id {chunk_id} not found.")

    def _get_path_from_pdf_id(self, pdf_id: str) -> str:
        """Get the full path from a PDF ID."""
        for path in self.paths:
            if self.pdfs[path]['pdf_id'] == pdf_id:
                return path
        return {'error': 'pdf_id not found.'}

    def _markdown_table_to_csv_json(self, md_text: str) -> list:
        """Convert markdown table to CSV/JSON format."""
        table_block = []
        capture = False
        for line in md_text.splitlines():
            if line.strip().startswith("|"):
                capture = True
                table_block.append(line)
            elif capture:
                break

        clean_lines = [re.sub(r'\s*\|\s*$', '', line.strip()) for line in table_block]
        clean_lines = [line.strip("|") for line in clean_lines if line.strip()]

        rows = []
        for line in clean_lines:
            parts = [re.sub(r'<br>', ' ', cell).strip() for cell in line.split("|")]
            rows.append(parts)

        if not rows:
            return []

        header = rows[0]
        rows = [r for r in rows[1:] if not all(set(c.strip()) <= {"-", ""} for c in r)]

        return [header] + rows

    def _setup_tools(self):
        """Set up the agent tools."""
        agent_self = self  # Reference to self for closures

        @tool
        def extract_text(pdf_id: str, query: str) -> str:
            """
            Retrieve relevant document context based on the user query.
            Args:
                pdf_id: The id of the PDF file.
                query: The user query.
            Returns:
                The retrieved document context.
            """
            pdf_filename = agent_self._get_path_from_pdf_id(pdf_id)
            if isinstance(pdf_filename, dict) and 'error' in pdf_filename:
                return json.dumps(pdf_filename)

            vectorstore = agent_self.pdfs[pdf_filename]['vectorstore']
            doc_render = agent_self.pdfs[pdf_filename]['doc_render']
            chunk_set = set()

            if vectorstore:
                compressor = FlashrankRerank()
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
                )
                docs = compression_retriever.invoke(query)[:4]
                document = []
                for doc in docs:
                    if doc.metadata.get('id') in chunk_set:
                        continue
                    chunk_set.add(doc.metadata.get('id'))
                    document.append(json.dumps({
                        "page": doc.metadata.get('page'),
                        "Retrieved": agent_self._find_chunk_by_id(
                            doc.metadata.get('id'), doc_render
                        )
                    }, indent=2))
                return "\n\n".join(document)
            return "No content available."

        @tool
        def extract_tables(pdf_id: str, page_num: int = -1) -> str:
            """
            Extract all tables from the rendered document for a specific page number.
            Args:
                pdf_id: The id of the PDF file.
                page_num: The page number to extract tables from. -1 for all pages.
            Returns:
                List of extracted tables as JSON string.
            """
            pdf_filename = agent_self._get_path_from_pdf_id(pdf_id)
            if isinstance(pdf_filename, dict) and 'error' in pdf_filename:
                return json.dumps(pdf_filename)

            doc_render = agent_self.pdfs[pdf_filename]['doc_render']
            tables = []
            chunks = doc_render.get('chunks', {})

            for _, content in chunks.items():
                if page_num != -1 and content.get('page') != page_num:
                    continue
                if content.get('block_type') != 'Table':
                    continue
                tables.append({
                    'page_number': content.get('page'),
                    'table_data': agent_self._markdown_table_to_csv_json(
                        content.get('markdown', '')
                    ),
                })

            return json.dumps(tables)

        @tool
        def extract_pdf_abstract(pdf_id: str) -> str:
            """
            Extract the abstract section from the PDF document.
            Args:
                pdf_id: The id of the PDF file.
            Returns:
                The extracted abstract text.
            """
            pdf_filename = agent_self._get_path_from_pdf_id(pdf_id)
            if isinstance(pdf_filename, dict) and 'error' in pdf_filename:
                return json.dumps(pdf_filename)

            if agent_self.pdfs[pdf_filename].get('abstract'):
                return agent_self.pdfs[pdf_filename]['abstract']

            doc_render = agent_self.pdfs[pdf_filename]['doc_render']
            page_renders = doc_render.get('page_renders', [])
            if not page_renders:
                return "No content available."

            markdown = page_renders[0].get('markdown', '')
            prompt = HumanMessage(
                content=f"Extract the abstract section from the following markdown text "
                        f"in not more than 200 words:\n\n{markdown}\n\n"
                        f"If no abstract is found, respond with 'No abstract found.'"
            )
            response = agent_self.llm.invoke([prompt])
            agent_self.pdfs[pdf_filename]['abstract'] = response.content.strip()
            return response.content.strip()

        @tool
        def extract_pdf_page(pdf_id: str, page_num: int) -> str:
            """
            Extract the text at a specific page from the PDF.
            Args:
                pdf_id: The id of the PDF file.
                page_num: The page number to extract.
            Returns:
                The extracted page text.
            """
            pdf_filename = agent_self._get_path_from_pdf_id(pdf_id)
            if isinstance(pdf_filename, dict) and 'error' in pdf_filename:
                return json.dumps(pdf_filename)

            doc_render = agent_self.pdfs[pdf_filename]['doc_render']
            page_renders = doc_render.get('page_renders', [])

            if page_num < 0 or page_num >= len(page_renders):
                return f"Page {page_num} not found. Available pages: 0-{len(page_renders)-1}"

            return page_renders[page_num].get('markdown', '')

        @tool
        def list_available_pdfs() -> str:
            """
            List all available PDF documents with their summaries.
            Returns:
                JSON string with PDF information.
            """
            pdf_info = [
                {
                    'pdf_id': agent_self.pdfs[path]['pdf_id'],
                    'summary': agent_self.pdfs[path]['summary']
                }
                for path in agent_self.paths
            ]
            return json.dumps(pdf_info, indent=2)

        self.tools = [extract_text, extract_tables, extract_pdf_abstract, extract_pdf_page, list_available_pdfs]
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.agent = self.llm.bind_tools(self.tools)

    def _update_pdf_info(self):
        """Update PDF info for system prompt."""
        self.pdf_info = [
            {
                'pdf_id': self.pdfs[path]['pdf_id'],
                'summary': self.pdfs[path]['summary']
            }
            for path in self.paths
        ]

    def _setup_graph(self):
        """Set up the LangGraph workflow."""

        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]

        agent_self = self

        def tool_node(state: AgentState):
            outputs = []
            for tool_call in state["messages"][-1].tool_calls:
                tool_result = agent_self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(ToolMessage(
                    content=json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                ))
            return {"messages": outputs}

        def call_model(state: AgentState, config: RunnableConfig):
            system_prompt = SystemMessage(
                "You are a helpful AI assistant that helps users understand information from PDF documents. "
                "Please respond to the user's query to the best of your ability!\n\n"
                "Following is information about the available PDF documents:\n\n"
                f"{json.dumps(agent_self.pdf_info, indent=2)}\n\n"
                "Use the PDF information to identify the relevant document and answer the user's query. "
                "If the user asks about a specific document, use the appropriate tools to retrieve information."
            )
            response = agent_self.agent.invoke([system_prompt] + list(state["messages"]), config)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if not last_message.tool_calls:
                return "end"
            return "continue"

        # Build the graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        self.graph = workflow.compile()

    def add_pdf(self, path: str, force_reload: bool = False):
        """
        Add a new PDF to the agent.

        Args:
            path: Path to the PDF file
            force_reload: If True, ignore cache and reprocess the PDF
        """
        if path not in self.paths:
            self.paths.append(path)

        print(f"Adding PDF: {Path(path).name}")

        # Check cache first (unless force_reload)
        cached_data = None
        if self.use_cache and self.cache and not force_reload:
            cached_data = self.cache.get(path, self.page_range)

        if cached_data:
            print(f"  [CACHED] Loading from cache")
            self.pdfs[path] = {
                'pdf_id': cached_data['pdf_id'],
                'converter': None,
                'doc_render': cached_data['doc_render'],
                'summary': cached_data['summary'],
                'abstract': cached_data.get('abstract'),
                'vectorstore': cached_data['vectorstore']
            }
        else:
            artifact_dict = create_model_dict()
            self._process_pdf(path, artifact_dict)

            # Cache the processed PDF
            if self.use_cache and self.cache and path in self.pdfs:
                self.cache.set(path, self.page_range, self.pdfs[path])
                print(f"  -> Cached for future use")

        self._update_pdf_info()
        print("PDF added successfully!")

    def chat(self, message: str, stream: bool = True) -> str:
        """
        Send a message to the agent and get a response.

        Args:
            message: The user message
            stream: Whether to stream the response

        Returns:
            The agent's response
        """
        inputs = {"messages": [("user", message)]}

        if stream:
            response_content = ""
            for s in self.graph.stream(inputs, stream_mode="values"):
                msg = s["messages"][-1]
                if hasattr(msg, 'content') and not hasattr(msg, 'tool_calls'):
                    response_content = msg.content
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"  [Using tool: {msg.tool_calls[0]['name']}]")
            return response_content
        else:
            result = self.graph.invoke(inputs)
            return result["messages"][-1].content

    def run_interactive(self):
        """Run the interactive chat interface."""
        print("\n" + "=" * 60)
        print("  PDF RAG Chat Agent - Interactive Mode")
        print("=" * 60)

        if not self.pdfs:
            print("\nNo PDFs loaded. Use '/add <path>' command to load PDFs.")
        else:
            print(f"\nLoaded {len(self.pdfs)} PDF document(s):")
            for path in self.paths:
                print(f"  - {self.pdfs[path]['pdf_id']}")

        cache_status = "enabled" if self.use_cache else "disabled"
        print(f"\nCache: {cache_status}")
        if self.use_cache and self.cache:
            print(f"Cache location: {self.cache.cache_dir}")

        print("\nCommands:")
        print("  /quit, /exit    - Exit the chat")
        print("  /list           - List available PDFs")
        print("  /add <path>     - Add a new PDF")
        print("  /reload <path>  - Reload PDF (ignore cache)")
        print("  /cache          - Show cache info")
        print("  /cache clear    - Clear all cached data")
        print("  /help           - Show this help message")
        print("-" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                    print("Goodbye!")
                    break

                if user_input.lower() == '/list':
                    if self.pdfs:
                        print("\nAvailable PDFs:")
                        for path in self.paths:
                            print(f"  - {self.pdfs[path]['pdf_id']}")
                            print(f"    Summary: {self.pdfs[path]['summary'][:100]}...")
                        print()
                    else:
                        print("No PDFs loaded.\n")
                    continue

                if user_input.lower().startswith('/add '):
                    pdf_path = user_input[5:].strip()
                    if os.path.exists(pdf_path):
                        self.add_pdf(pdf_path)
                    else:
                        print(f"File not found: {pdf_path}\n")
                    continue

                if user_input.lower().startswith('/reload '):
                    pdf_path = user_input[8:].strip()
                    if os.path.exists(pdf_path):
                        self.add_pdf(pdf_path, force_reload=True)
                    else:
                        print(f"File not found: {pdf_path}\n")
                    continue

                if user_input.lower() == '/cache':
                    if self.use_cache and self.cache:
                        cached = self.cache.list_cached()
                        print(f"\nCache location: {self.cache.cache_dir}")
                        print(f"Cached PDFs: {len(cached)}")
                        for item in cached:
                            print(f"  - {item['pdf_id']} (pages: {item['page_range']})")
                            print(f"    Cached at: {item['cached_at']}")
                        print()
                    else:
                        print("Caching is disabled.\n")
                    continue

                if user_input.lower() == '/cache clear':
                    if self.use_cache and self.cache:
                        confirm = input("Are you sure you want to clear all cached data? (y/N): ")
                        if confirm.lower() == 'y':
                            self.cache.clear()
                        else:
                            print("Cache clear cancelled.\n")
                    else:
                        print("Caching is disabled.\n")
                    continue

                if user_input.lower() == '/help':
                    print("\nCommands:")
                    print("  /quit, /exit    - Exit the chat")
                    print("  /list           - List available PDFs")
                    print("  /add <path>     - Add a new PDF")
                    print("  /reload <path>  - Reload PDF (ignore cache)")
                    print("  /cache          - Show cache info")
                    print("  /cache clear    - Clear all cached data")
                    print("  /help           - Show this help message")
                    print("\nYou can ask questions about the loaded PDF documents.\n")
                    continue

                # Process regular chat message
                print("\nAssistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDF RAG Chat Agent - Interactive document Q&A"
    )
    parser.add_argument(
        '--pdf-dir', '-d',
        type=str,
        default=None,
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--pdf', '-p',
        type=str,
        action='append',
        default=[],
        help='Path to a PDF file (can be used multiple times)'
    )
    parser.add_argument(
        '--page-range', '-r',
        type=str,
        default='0-9',
        help='Page range to process (e.g., "0-9" or "0,1,2,5-10")'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of processed PDFs'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Custom cache directory (default: ~/.cache/rag_chat)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cached data and exit'
    )

    args = parser.parse_args()

    # Handle clear-cache command
    if args.clear_cache:
        cache = PDFCache(Path(args.cache_dir) if args.cache_dir else None)
        cache.clear()
        return

    # Parse page range
    page_range = []
    for part in args.page_range.split(','):
        if '-' in part:
            start, end = part.split('-')
            page_range.extend(range(int(start), int(end) + 1))
        else:
            page_range.append(int(part))

    # Determine PDF paths
    pdf_paths = args.pdf if args.pdf else None

    # Create and run agent
    agent = PDFRAGAgent(
        pdf_paths=pdf_paths,
        pdf_dir=args.pdf_dir,
        page_range=page_range,
        model_name=args.model,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )

    # Load PDFs if any are specified
    if agent.paths:
        agent.load_pdfs()

    # Run interactive mode
    agent.run_interactive()


if __name__ == "__main__":
    main()
