from typing import Optional

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class SectionHeader(Block):
    block_type: BlockTypes = BlockTypes.SectionHeader
    heading_level: Optional[int] = None
    block_description: str = "The header of a section of text or other blocks."
    html: str | None = None

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.ignore_for_output:
            return ""

        if self.html:
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        template = template.replace("\n", " ")
        tag = f"h{self.heading_level}" if self.heading_level else "h2"
        return f"<{tag}>{template}</{tag}>"

    def verify(self, document, page) -> bool:
        from marker.schema.document import Document
        from marker.schema.groups import PageGroup
        document: Document = document
        page: PageGroup = page
        '''
        Verify if this block is indeed a SectionHeader.
        '''
        for chunks in document.get_all_chunks():
            '''
            text - chunk.raw_text()
            height - block.line_height(document)
            page_height - page.polygon.height
            bbox - chunk.polygon.bbox - (x0, y0, x1, y1)
            chunk.id = str(chunk.id)
            '''
            raise NotImplementedError("SectionHeader verification not implemented yet.")

    def convert_to_page_header(self):
        from marker.schema.blocks import PageHeader
        page_header = PageHeader(
            polygon=self.polygon,
            block_id=self.block_id,
            page_id=self.page_id,
            text_extraction_method=self.text_extraction_method,
            structure=self.structure,
            replace_output_newlines=False,
            ignore_for_output=True,
            source=self.source,
            top_k={BlockTypes.PageHeader: 1.0},
            metadata=self.metadata,
            lowres_image=self.lowres_image,
            highres_image=self.highres_image,
            removed=self.removed,
            heading_level=None,
            html=self.html
        )
        return page_header
