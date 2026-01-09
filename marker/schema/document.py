from __future__ import annotations

from typing import List, Sequence, Optional

from pydantic import BaseModel

import numpy as np
import re
import math
from collections import Counter

from marker.schema import BlockTypes
from marker.schema.blocks import Block, BlockId, BlockOutput
from marker.schema.groups.page import PageGroup


class DocumentOutput(BaseModel):
    children: List[BlockOutput]
    html: str
    block_type: BlockTypes = BlockTypes.Document


class TocItem(BaseModel):
    title: str
    heading_level: int
    page_id: int
    polygon: List[List[float]]


def levenshtein_distance(s1, s2):
    """
    Calculates the Levenshtein distance between two NORMALIZED strings.
    """
    # Normalize strings first to ignore whitespace and symbols
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text
    
    s1 = normalize_text(s1)
    s2 = normalize_text(s2)

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def lexical_similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """Calculate lexical similarity between two texts."""
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.split()
    
    tokens1, tokens2 = tokenize(text1), tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0

    set1, set2 = set(tokens1), set(tokens2)

    if method == "overlap":
        intersection = len(set1 & set2)
        smaller = min(len(set1), len(set2))
        return intersection / smaller if smaller else 0.0
    elif method == "cosine":
        freq1, freq2 = Counter(tokens1), Counter(tokens2)
        all_tokens = set(freq1) | set(freq2)
        dot = sum(freq1[t] * freq2[t] for t in all_tokens)
        mag1 = math.sqrt(sum(v ** 2 for v in freq1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in freq2.values()))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    else:
        raise ValueError("Invalid method for lexical similarity.")

def heading_similarity(text1: str, text2: str) -> bool:
    """
    Determines if two headings are similar using a two-path check.
    """
    overlap_score = lexical_similarity(text1, text2, method="overlap")

    # Method 1: Original strict check
    if overlap_score > 0.99:
        cosine_score = lexical_similarity(text1, text2, method="cosine")
        if cosine_score > 0.9:
            return True

    # Method 2: New lenient check for max two character errors
    if overlap_score > 0.5:
        edit_distance = levenshtein_distance(text1, text2)
        if edit_distance < 3:
            return True
    
    return False


class Document(BaseModel):
    filepath: str
    pages: List[PageGroup]
    block_type: BlockTypes = BlockTypes.Document
    table_of_contents: List[TocItem] | None = None
    debug_data_path: str | None = None  # Path that debug data was saved to
    exclude_list: List[BlockTypes] = [BlockTypes.Span, BlockTypes.Line]  # List of block types to exclude from output

    def get_block(self, block_id: BlockId):
        page = self.get_page(block_id.page_id)
        block = page.get_block(block_id)
        if block:
            return block
        return None

    def get_page(self, page_id):
        for page in self.pages:
            if page.page_id == page_id:
                return page
        return None

    def get_next_block(
        self, block: Block, ignored_block_types: List[BlockTypes] = None
    ):
        if ignored_block_types is None:
            ignored_block_types = []
        next_block = None

        # Try to find the next block in the current page
        page = self.get_page(block.page_id)
        next_block = page.get_next_block(block, ignored_block_types)
        if next_block:
            return next_block

        # If no block found, search subsequent pages
        for page in self.pages[self.pages.index(page) + 1 :]:
            next_block = page.get_next_block(None, ignored_block_types)
            if next_block:
                return next_block
        return None

    def get_next_page(self, page: PageGroup):
        page_idx = self.pages.index(page)
        if page_idx + 1 < len(self.pages):
            return self.pages[page_idx + 1]
        return None

    def get_prev_block(self, block: Block):
        page = self.get_page(block.page_id)
        prev_block = page.get_prev_block(block)
        if prev_block:
            return prev_block
        prev_page = self.get_prev_page(page)
        if not prev_page:
            return None
        return prev_page.get_block(prev_page.structure[-1])

    def get_prev_page(self, page: PageGroup):
        page_idx = self.pages.index(page)
        if page_idx > 0:
            return self.pages[page_idx - 1]
        return None

    def assemble_html(
        self, child_blocks: List[Block], block_config: Optional[dict] = None
    ):
        template = ""
        for c in child_blocks:
            template += f"<content-ref src='{c.id}'></content-ref>"
        return template

    def render(self, block_config: Optional[dict] = None):
        child_content = []
        section_hierarchy = None
        for page in self.pages:
            rendered = page.render(self, None, section_hierarchy, block_config)
            section_hierarchy = rendered.section_hierarchy.copy()
            child_content.append(rendered)

        out = DocumentOutput(
            children=child_content,
            html=self.assemble_html(child_content, block_config),
        )
        return out

    def contained_blocks(self, block_types: Sequence[BlockTypes] = None) -> List[Block]:
        blocks = []
        for page in self.pages:
            blocks += page.contained_blocks(self, block_types)
        return blocks
    

    def verify_headers(self) -> list:

        def core_logic(chunks, page_height):
            """
            Core logic to determine if chunks should be reclassified as PageHeaders.
            Returns True if chunks should be reclassified as PageHeaders.
            """
            max_threshold = 0.0009
            top_portion_limit = page_height * 0.1
            # Check if all chunks are in top portion of pages
            for chunk in chunks:
                if chunk['bbox'][1] > top_portion_limit:  # y0 coordinate
                    return False
            # Calculate standard deviation of top margins 
            top_margins = [chunk['bbox'][1] for chunk in chunks]
            if len(top_margins) > 1:
                std_dev_margin = np.std(top_margins, ddof=1)
                threshold = std_dev_margin / page_height
                return threshold <= max_threshold
            return False

        """
        Returns list of chunk IDs that should be PageHeader.
        """
        # print("Verifying SectionHeaders for PageHeader reclassification...")
        page_height = self.pages[0].polygon.height if self.pages else 0

        section_headers = self.get_all_chunks([BlockTypes.SectionHeader])
        # Group similar section headers
        text_groups = []
        for chunk in section_headers:
            text = chunk.raw_text(self).strip()
            if not text:
                continue
                
            bbox = chunk.polygon.bbox  # (x0, y0, x1, y1)
            chunk_data = {
                "id": (chunk.id),
                "chunk": chunk,
                "bbox": bbox,
                "text": text
            }
            
            # Find existing group or create new one
            found_group = False
            for group in text_groups:
                representative_text = group[0]['text']
                if heading_similarity(text, representative_text):
                    group.append(chunk_data)
                    found_group = True
                    break
            
            if not found_group:
                text_groups.append([chunk_data])
        
        # Collect chunk IDs that need reclassification
        reclassification_ids = []
        
        for group in text_groups:
            if len(group) > 1:
                if page_height and core_logic(group, page_height):
                    # Add all chunk IDs from this group to reclassification list
                    for chunk_data in group:
                        block = self.get_block(chunk_data['id'])
                        # print("Converting SectionHeader to PageHeader:", chunk_data['id'], block.raw_text(self))
                        newblock = block.convert_to_page_header()
                        self.get_page(block.page_id).replace_block(block, newblock)

        # return reclassification_ids


    def get_all_chunks(self, block_type: list[BlockTypes] = []) -> List[Block]:
        all_chunks = []
        for page in self.pages:
            for block_id in page.structure:
                if block_id in self.exclude_list:
                    continue
                block = page.get_block(block_id)
                if block_type == [] or block.block_type in block_type:
                    all_chunks.append(block)
        return all_chunks
