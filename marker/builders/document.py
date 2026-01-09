from typing import Annotated, List

from marker.builders import BaseBuilder
from marker.builders.layout import LayoutBuilder
from marker.builders.line import LineBuilder
from marker.builders.ocr import OcrBuilder
from marker.providers.pdf import PdfProvider
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.registry import get_block_class


class DocumentBuilder(BaseBuilder):
    """
    Constructs a Document given a PdfProvider, LayoutBuilder, and OcrBuilder.
    """
    lowres_image_dpi: Annotated[
        int,
        "DPI setting for low-resolution page images used for Layout and Line Detection.",
    ] = 96
    highres_image_dpi: Annotated[
        int,
        "DPI setting for high-resolution page images used for OCR.",
    ] = 192
    disable_ocr: Annotated[
        bool,
        "Disable OCR processing.",
    ] = True

    def __init__(self, config = None, **kwargs):
        super().__init__(config)
        if 'ignore_blocks' in kwargs:
            self.ignore_blocks = [BlockTypes[blk] for blk in kwargs['ignore_blocks']]
        else:
            self.ignore_blocks = []
        
        if config.get('ignore_before_TOC', False):
            self.ignore_before_TOC = config['ignore_before_TOC']

    def __call__(self, provider: PdfProvider, layout_builder: LayoutBuilder, line_builder: LineBuilder, ocr_builder: OcrBuilder):
        document = self.build_document(provider)
        layout_builder(document, provider)
        line_builder(document, provider)
        if self.ignore_blocks:
            self.init_ignore(document.pages)

        if not self.disable_ocr:
            ocr_builder(document, provider)
        return document

    def init_ignore(self, pages: List[PageGroup]):
        # Ignore blocks
        for i, page in enumerate(pages):
            for block_id in page.structure:
                block = page.get_block(block_id)
                if block.block_type in self.ignore_blocks:
                    block.ignore_for_output = True
                    # In case of Table of Contents ignore the Page
                    if block.block_type == BlockTypes.TableOfContents and page.page_id <= 10:
                        if self.ignore_before_TOC:
                            for j in range(i):
                                pages[j].ignore_for_output = True
                        page.ignore_for_output = True
                    break
        # Ignore blocks
        for i, page in enumerate(pages):
            if page.ignore_for_output:
                for block in page.children:
                    block.ignore_for_output = True

    def build_document(self, provider: PdfProvider):
        PageGroupClass: PageGroup = get_block_class(BlockTypes.Page)
        lowres_images = provider.get_images(provider.page_range, self.lowres_image_dpi)
        highres_images = provider.get_images(provider.page_range, self.highres_image_dpi)
        initial_pages = [
            PageGroupClass(
                page_id=p,
                lowres_image=lowres_images[i],
                highres_image=highres_images[i],
                polygon=provider.get_page_bbox(p),
                refs=provider.get_page_refs(p)
            ) for i, p in enumerate(provider.page_range)
        ]
        DocumentClass: Document = get_block_class(BlockTypes.Document)
        return DocumentClass(filepath=provider.filepath, pages=initial_pages)
