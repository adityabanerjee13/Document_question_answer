import csv
import os
import re
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disables a tokenizers warning

from collections import defaultdict
from typing import Annotated, Any, Dict, List, Optional, Type, Tuple, Union
import io
from contextlib import contextmanager
import tempfile

from marker.processors import BaseProcessor
from marker.services import BaseService
from marker.processors.llm.llm_table_merge import LLMTableMergeProcessor
from marker.providers.registry import provider_from_filepath
from marker.renderers.chunk import ChunkRenderer
from marker.builders.document import DocumentBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.builders.structure import StructureBuilder
from marker.converters import BaseConverter
from marker.processors.blockquote import BlockquoteProcessor
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.list import ListProcessor
from marker.processors.llm.llm_complex import LLMComplexRegionProcessor
from marker.processors.llm.llm_form import LLMFormProcessor
from marker.processors.llm.llm_image_description import LLMImageDescriptionProcessor
from marker.processors.llm.llm_table import LLMTableProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.reference import ReferenceProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor
from marker.processors.block_relabel import BlockRelabelProcessor
from marker.processors.blank_page import BlankPageProcessor
from marker.processors.llm.llm_equation import LLMEquationProcessor
from marker.renderers.page_markdown import PageMarkdownRenderer
from marker.renderers.markdown import MarkdownRenderer, MarkdownOutput
from marker.renderers.markdown import cleanup_text
from marker.renderers import BaseRenderer
from marker.schema.document import Document
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.registry import register_block_class
from marker.util import strings_to_classes
from marker.processors.llm.llm_handwriting import LLMHandwritingProcessor
from marker.processors.order import OrderProcessor
from marker.services.gemini import GoogleGeminiService
from marker.processors.line_merge import LineMergeProcessor
from marker.processors.llm.llm_mathblock import LLMMathBlockProcessor
from marker.processors.llm.llm_page_correction import LLMPageCorrectionProcessor
from marker.processors.llm.llm_sectionheader import LLMSectionHeaderProcessor



def renderer2cls_loc(renderer: str) -> Optional[str|List[str]]:
    if renderer == 'pageMarkdown':
        renderer = 'marker.renderers.page_markdown.PageMarkdownRenderer'
    elif renderer == 'markdown':
        renderer = 'marker.renderers.markdown.MarkdownRenderer'
    elif renderer == 'chunks':
        renderer = 'marker.renderers.chunk.ChunkRenderer'
    elif renderer == 'json':
        renderer = 'marker.renderers.json.JSONRenderer'
    elif renderer == 'html':
        renderer = 'marker.renderers.html.HTMLRenderer'
    elif '+' in renderer:
        renderers = renderer.split('+')
        renderer = [renderer2cls_loc(r.strip()) for r in renderers]
    else:
        renderer = None
    return renderer

def markdown_to_hierarchical_json(md_text):
    lines = md_text.strip().splitlines()

    # Root structure to hold everything
    root = {"type": "root", "children": []}
    stack = [root]  # stack to keep track of heading hierarchy
    table_block = []  # to store detected table lines
    in_table = False  # track table region

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Detect start of table (line starts with |)
        if line.strip().startswith("|"):
            in_table = True
            table_block.append(line)
            continue
        elif in_table and not line.strip().startswith("|"):
            # end of table
            csv_text = convert_table_to_csv(table_block)
            table_node = {"type": "table", "csv": csv_text}
            stack[-1]["children"].append(table_node)
            table_block = []
            in_table = False

        if in_table:
            continue  # skip rest of loop while collecting table lines

        # Headings: #, ##, ###, etc.
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            node = {
                "type": "section",
                "level": level,
                "heading": heading_text,
                "children": []
            }

            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()

            stack[-1]["children"].append(node)
            stack.append(node)
            continue

        # Lists: - item
        list_match = re.match(r'^\s*[-*+]\s+(.*)', line)
        if list_match:
            item_text = list_match.group(1).strip()
            parent = stack[-1]
            if not parent["children"] or parent["children"][-1]["type"] != "list":
                parent["children"].append({"type": "list", "items": []})
            parent["children"][-1]["items"].append(item_text)
            continue

        # Regular paragraph
        parent = stack[-1]
        parent["children"].append({
            "type": "paragraph",
            "text": line.strip()
        })

    # Handle case where file ends with table
    if table_block:
        csv_text = convert_table_to_csv(table_block)
        table_node = {"type": "table", "csv": csv_text}
        stack[-1]["children"].append(table_node)

    return root["children"]

def convert_table_to_csv(table_lines):
    """Convert a Markdown table block into CSV text"""
    clean_lines = [re.sub(r'\s*\|\s*$', '', line.strip()) for line in table_lines]
    clean_lines = [line.strip("|") for line in clean_lines if line.strip()]

    rows = []
    for line in clean_lines:
        parts = [re.sub(r'<br>', ' ', cell).strip() for cell in line.split("|")]
        rows.append(parts)

    # Remove separator (---|---|---)
    rows = [r for r in rows if not all(set(c.strip()) <= {"-", ""} for c in r)]
    if not rows:
        return ""

    header = rows[0]
    data_rows = rows[1:]

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(header)
    for r in data_rows:
        writer.writerow(r)
    return csv_buffer.getvalue()

class PdfConverter(BaseConverter):
    """
    A converter for processing and rendering PDF files into Markdown, JSON, HTML and other formats.
    """

    override_map: Annotated[
        Dict[BlockTypes, Type[Block]],
        "A mapping to override the default block classes for specific block types.",
        "The keys are `BlockTypes` enum values, representing the types of blocks,",
        "and the values are corresponding `Block` class implementations to use",
        "instead of the defaults.",
    ] = defaultdict()
    use_llm: Annotated[
        bool,
        "Enable higher quality processing with LLMs.",
    ] = False
    
    default_processors: Tuple[BaseProcessor, ...] = (
        OrderProcessor,
        BlockRelabelProcessor,
        LineMergeProcessor,
        BlockquoteProcessor,
        CodeProcessor,
        EquationProcessor,
        FootnoteProcessor,
        IgnoreTextProcessor,
        LineNumbersProcessor,
        ListProcessor,
        DocumentTOCProcessor,
        PageHeaderProcessor,
        SectionHeaderProcessor,
        TableProcessor,
        # LLMTableProcessor,
        # LLMTableMergeProcessor,
        # LLMFormProcessor,
        TextProcessor,
        # LLMComplexRegionProcessor,
        # LLMImageDescriptionProcessor,
        # LLMEquationProcessor,
        # LLMHandwritingProcessor,
        # LLMMathBlockProcessor,
        LLMSectionHeaderProcessor,
        # LLMPageCorrectionProcessor,
        ReferenceProcessor,
        BlankPageProcessor,
        DebugProcessor,
    )
    default_llm_service: BaseService = GoogleGeminiService

    def __init__(
        self,
        artifact_dict: Dict[str, Any],
        processor_list: Optional[List[str]] = None,
        config=None,
    ):
        
        renderer = config.get("renderer", None)
        renderer = renderer2cls_loc(renderer)
        # remove renderer from config to avoid issues in other places
        if "renderer" in config:
            config.pop("renderer")
        
        llm_service = config.get("llm_service", None)
        # remove llm_service from config to avoid issues in other places
        if "llm_service" in config:
            config.pop("llm_service")

        super().__init__(config)

        if config is None:
            config = {}

        # Block types to ignore are initialized here.
        if config.get("ignore_TOC", False):
            self.ignore_blocks = ['TableOfContents']
        else:
            self.ignore_blocks = []

        for block_type, override_block_type in self.override_map.items():
            register_block_class(block_type, override_block_type)

        if processor_list is not None:
            processor_list = strings_to_classes(processor_list)
        else:
            processor_list = self.default_processors

        if renderer:
            renderer = strings_to_classes([renderer] if isinstance(renderer, str) else renderer)
        else:
            renderer = [PageMarkdownRenderer, ChunkRenderer]

        # Put here so that resolve_dependencies can access it
        self.artifact_dict = artifact_dict

        if llm_service:
            llm_service_cls = strings_to_classes([llm_service])[0]
            llm_service = self.resolve_dependencies(llm_service_cls)
        elif config.get("use_llm", False):
            llm_service = self.resolve_dependencies(self.default_llm_service)

        # Inject llm service into artifact_dict so it can be picked up by processors, etc.
        self.artifact_dict["llm_service"] = llm_service
        self.llm_service = llm_service

        self.renderer = renderer

        processor_list = self.initialize_processors(processor_list)
        self.processor_list = processor_list

        self.layout_builder_class = LayoutBuilder
        self.page_count = None  # Track how many pages were converted

    @contextmanager
    def filepath_to_str(self, file_input: Union[str, io.BytesIO]):
        temp_file = None
        try:
            if isinstance(file_input, str):
                yield file_input
            else:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    if isinstance(file_input, io.BytesIO):
                        file_input.seek(0)
                        temp_file.write(file_input.getvalue())
                    else:
                        raise TypeError(
                            f"Expected str or BytesIO, got {type(file_input)}"
                        )

                yield temp_file.name
        finally:
            if temp_file is not None and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def build_document(self, filepath: str):
        provider_cls = provider_from_filepath(filepath)
        layout_builder = self.resolve_dependencies(self.layout_builder_class)
        line_builder = self.resolve_dependencies(LineBuilder)
        ocr_builder = self.resolve_dependencies(OcrBuilder)
        provider = provider_cls(filepath, self.config)
        document = DocumentBuilder(self.config, ignore_blocks=self.ignore_blocks)(
            provider, layout_builder, line_builder, ocr_builder
        )
        structure_builder_cls = self.resolve_dependencies(StructureBuilder)
        structure_builder_cls(document)

        for processor in self.processor_list:
            processor(document)

        return document

    def render(self, renderer_cls: Type[BaseRenderer], document: Document) -> Tuple[str, Any, Dict[str, Any]]:
        renderer = self.resolve_dependencies(renderer_cls)
        if renderer_cls.__name__ == "PageMarkdownRenderer":
            page_output, images, metadata = renderer(document)
            return "page_renders", page_output, images, metadata

        elif renderer_cls.__name__ == "ChunkRenderer":
            json_output, images, metadata = renderer(document)
            return "chunks", json_output, images, metadata

        elif renderer_cls.__name__ == "MarkdownRenderer":
            rendered, images, metadata = renderer(document)
            if isinstance(rendered, str):
                rendered = cleanup_text(rendered)
            return "markdown", rendered, images, metadata
        
        elif renderer_cls.__name__ == "JSONRenderer":
            renderer = self.resolve_dependencies(MarkdownRenderer)
            out = renderer(document)
            if isinstance(out, MarkdownOutput):
                md = out.markdown
                images = out.images
                metadata = out.metadata
            elif isinstance(out, dict) and 'markdown' in out:
                md = out['markdown']
                images = out.get('images', {})
                metadata = out.get('metadata', {})
            elif isinstance(out, tuple) and len(out) == 3:
                md, images, metadata = out
            else:
                raise ValueError("Unexpected output from MarkdownRenderer for JSON conversion.")
            md_json = markdown_to_hierarchical_json(md)
            return 'json', md_json, images, metadata
        
        else:
            raise ValueError(f"Unsupported renderer class: {renderer_cls.__name__}")

    def render_document(self, document: Document) -> Dict[str, Any]:
        out_render = {}
        out_render['page_structure'] = {}
        for i, doc_child in enumerate(document.pages):
            if doc_child.ignore_for_output:
                out_render['page_structure'][doc_child.page_id] = []
            else:
                out_render['page_structure'][doc_child.page_id] = [str(identity) for identity in doc_child.structure]

        if isinstance(self.renderer, list):
            for j, renderer_cls in enumerate(self.renderer):
                key, rendered, images, metadata = self.render(renderer_cls, document)
                out_render[key] = rendered
                out_render['images'] = images
            out_render['metadata'] = metadata

        elif issubclass(self.renderer, BaseRenderer):
            key, rendered, images, metadata = self.render(self.renderer, document)
            out_render[key] = rendered
            out_render['metadata'] = metadata
            out_render['images'] = images

        else:
            raise ValueError("Renderer must be a BaseRenderer subclass or a list of BaseRenderer subclasses.")

        return out_render

    def __call__(self, filepath: str | io.BytesIO):
        with self.filepath_to_str(filepath) as temp_path:
            document = self.build_document(temp_path)
            self.page_count = len(document.pages)
            out_render = self.render_document(document)
        return out_render, document
