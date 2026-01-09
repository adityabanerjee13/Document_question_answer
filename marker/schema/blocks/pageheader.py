from marker.schema import BlockTypes
from marker.schema.blocks import Block

# from marker.schema.groups.page import PageGroup
# from marker.schema.document import Document 

import numpy as np

class PageHeader(Block):
    block_type: BlockTypes = BlockTypes.PageHeader
    block_description: str = (
        "Text that appears at the top of a page, like a page title."
    )
    replace_output_newlines: bool = True
    ignore_for_output: bool = True
    html: str | None = None
    verifying_list: list = [BlockTypes.Text, BlockTypes.SectionHeader]

    def assemble_html(self, document, child_blocks, parent_structure, block_config):
        if block_config and block_config.get("keep_pageheader_in_output"):
            self.ignore_for_output = False

        if self.html and not self.ignore_for_output:
            return self.html

        return super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )

    def verify(self, document, page) -> bool:
        page_height = page.polygon.height
        if self.polygon.bbox[1] > 0.2 * page_height:
            return False
        header_height = self.line_height(document)
        text_heights = []
        for block in page.children:
            if block.block_type in [BlockTypes.Text]:
                block_height = block.line_height(document)
                text_heights.append(block_height)
                
            if block.block_type in [BlockTypes.SectionHeader]:
                block_height = block.line_height(document)
                if block_height < header_height:
                    return False
                elif abs(block_height - header_height) < 0.01*block_height:
                    return False
                
            # if block.block_type in [BlockTypes.PageFooter]:
            #     block_height = block.line_height(document)
            #     if block_height and abs(block_height-header_height) > 0.01*block_height:
            #         return True
            
        if header_height <= np.mean(text_heights):
            return True
        
        return True

    def convert_to_sectionheader(self):
        from marker.schema.blocks import SectionHeader
        section_header = SectionHeader(
            polygon=self.polygon,
            block_id=self.block_id,
            page_id=self.page_id,
            text_extraction_method=self.text_extraction_method,
            structure=self.structure,
            replace_output_newlines=False,
            ignore_for_output=False,
            source=self.source,
            top_k={BlockTypes.SectionHeader: 1.0},
            metadata=self.metadata,
            lowres_image=self.lowres_image,
            highres_image=self.highres_image,
            removed=self.removed,
            doc_toc_level=self.doc_toc_level,
            heading_level=None,
            html=self.html
        )
        return section_header

        # def core_logic(chunks):
        #     # Check if all chunks are in the top 10% of the page.
        #     top_portion_limit = PAGE_HEIGHT * 0.1
        #     for chunk in chunks:
        #         if chunk['bbox'][1] > top_portion_limit:
        #             return False  

        #     # if std of location big
        #     top_margins = [chunk['bbox'][1] for chunk in chunks]
        #     std_dev_margin = np.std(top_margins, axis=0)
        #     threshold = std_dev_margin / PAGE_HEIGHT
        #     return threshold <= max_threshold
        

        return False
    

    def resolve_header(self, bbox, layout_result):
        # resolving page header issue
        # compare bbox height with (SectionHeader,Text) height
        header_height = bbox.height
        for other_bbox in layout_result.bboxes:
            if other_bbox.label in ['SectionHeader'] and other_bbox.height < header_height:
                # likely not a page header
                bbox.label = 'SectionHeader'
                bbox.top_k = {'SectionHeader': 1, 'PageHeader': 0, 'ListItem': 0, 'Text': 0, 'Picture': 0}
                bbox.confidence = 1
                break
            elif other_bbox.label in ['SectionHeader'] and abs(other_bbox.height - header_height) < 0.05*other_bbox.height:
                # likely not a page header
                bbox.label = 'SectionHeader'
                bbox.top_k = {'SectionHeader': 1, 'PageHeader': 0, 'ListItem': 0, 'Text': 0, 'Picture': 0}
                bbox.confidence = 1
                break
            if other_bbox.label in ['PageFooter'] and abs(other_bbox.height-header_height) > 0.05*other_bbox.height:
                # likely not a page header
                bbox.label = 'SectionHeader'
                bbox.top_k = {'SectionHeader': 1, 'PageHeader': 0, 'ListItem': 0, 'Text': 0, 'Picture': 0}
                bbox.confidence = 1
                break
