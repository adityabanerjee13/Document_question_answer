from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
import pypdfium2 as pdfium
import fitz
import numpy as np
# from bs4 import BeautifulSoup
import re
from collections import Counter
import math


# def HTML2Text(html):
#     if not html:
#         return ""
#     soup = BeautifulSoup(html, "html.parser")
#     text = soup.get_text()  # \n for <br>, etc.
#     return text

def lexical_similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """
    Calculate lexical similarity between two texts.
    
    Supported methods:
        - 'jaccard': Jaccard index based on unique tokens
        - 'cosine': Cosine similarity based on token frequencies
        - 'overlap': Overlap coefficient based on token intersection
    """
    
    # --- Normalize & tokenize ---
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return text.split()
    
    tokens1, tokens2 = tokenize(text1), tokenize(text2)
    
    if not tokens1 or not tokens2:
        return 0.0

    # --- Compute similarity ---
    set1, set2 = set(tokens1), set(tokens2)
    
    if method == "jaccard":
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0

    elif method == "overlap":
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
        raise ValueError("Invalid method for lexical similarity. Choose from ['jaccard', 'cosine', 'overlap'].")

# check if 2 bbox overlap
def isoverlapping(box1, box2):
    if not (box1 and box2):
        return False
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    # Check if one rectangle is to the left of the other
    if x0_1 >= x1_2 or x0_2 >= x1_1:
        return False

    # Check if one rectangle is above the other
    if y0_1 >= y1_2 or y0_2 >= y1_1:
        return False

    return True

# bbox similarity score intersection over union
def bbox_IoU(box1, box2):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    xi0 = max(x0_1, x0_2)
    yi0 = max(y0_1, y0_2)
    xi1 = min(x1_1, x1_2)
    yi1 = min(y1_1, y1_2)

    inter_area = max(0, xi1 - xi0) * max(0, yi1 - yi0)

    box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
    box2_area = (x1_2 - x0_2) * (y1_2 - y0_2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area else 0.0

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



class DocumentTOCProcessor(BaseProcessor):
    """
    A processor for generating a table of contents for the document.
    """
    block_types = (BlockTypes.SectionHeader, )

    def walk_outline(self, outline_item, toc_info: list, parent_level=0):
        """Recursively walk through PDF outline/bookmarks with enhanced level detection."""
        title = outline_item.title
        page_index = outline_item.page_index
        
        # Detect level
        detected_level = outline_item.level
        
        # Use detected level if available, otherwise use parent_level + 1
        if detected_level:
            level = detected_level + 1
        else:
            level = parent_level + 1
        
        toc_info.append({'title': title, 'level': level, 'page_index': page_index})

        # Process children if any
        if hasattr(outline_item, 'children') and outline_item.children:
            for child in outline_item.children:
                self.walk_outline(child, toc_info, level)

    def get_document_toc(self, path):
        pdf = pdfium.PdfDocument(path)
        toc = pdf.get_toc()

        toc_info = []
        if toc:
            for item in toc:
                self.walk_outline(item, toc_info)

        return toc_info
    
    def merge_document_toc(self, document: Document, toc_info):
        
        def get_overlapping_texts(title, info_obj, type="page"):
            if type == "page":
                blocks = info_obj.get_text_blocks()
                overlapping_texts = []
                for x0, y0, x1, y1, text, _, _ in blocks:
                    block_bbox = [x0, y0, x1, y1]
                    score = lexical_similarity(text.strip(), title.strip(), method="overlap")
                    if score > 0.9:
                        overlapping_texts.append((text, block_bbox))
                return overlapping_texts
            else:
                out = []
                page = document.get_page(info_obj)
                for block in page.contained_blocks(document, list(self.block_types)):
                    text = block.raw_text(document).strip()

                    if not text: # empty text
                        continue
                    if abs(len(text)-len(title))>60: # 
                        continue
                    score = lexical_similarity(text, title, method="overlap")
                    if score > 0.9:
                        out.append(block.id)
                return out

        doc = fitz.open(document.filepath)

        for info in toc_info:
            title = info['title']
            level = info['level']
            page_index = info['page_index']
            if not document.get_page(page_index):
                continue

            page_obj = doc[page_index]

            text_blocks = get_overlapping_texts(title, page_obj)

            if text_blocks:
                cosine_scores = []
                jaccard_scores = []
                for i, (text, block_bbox) in enumerate(text_blocks):
                    cosine_scores.append(lexical_similarity(text, title, method="cosine"))
                    jaccard_scores.append(lexical_similarity(text, title, method="jaccard"))
                
                cosine_scores = np.array(cosine_scores)
                jaccard_scores = np.array(jaccard_scores)
                if len(cosine_scores) > 0:
                    cosine_scores = softmax(cosine_scores)

                if len(jaccard_scores) > 0:
                    jaccard_scores = softmax(jaccard_scores)
                combined_scores = (cosine_scores + jaccard_scores) / 2 if len(cosine_scores) > 0 else []
                
                max_prob = np.argmax(combined_scores)

                title_text, title_bbox = text_blocks[max_prob]
            else: title_text, title_bbox = None, None
            
            overlapping_blocks = get_overlapping_texts(title, page_index, type="document")
            
            if len(overlapping_blocks)<=1:
                if overlapping_blocks:
                    block = document.get_block(overlapping_blocks[0])
                    block.heading_level = level
                    block.doc_toc_level = True
                else: continue

            cosine_scores = []
            jaccard_scores = []
            IoU_break = False
            for block_id in overlapping_blocks:
                block = document.get_block(block_id)
                bbox = block.polygon.bbox
                if isoverlapping(title_bbox, bbox):
                    if bbox_IoU(title_bbox, bbox)>0.9:
                        block.heading_level = level 
                        block.doc_toc_level = True
                        IoU_break = True
                        break
                    pass
                text = block.raw_text(document).strip()
                comp_text = title_text if title_text and lexical_similarity(title_text, text) > lexical_similarity(title, text) else title

                cosine_scores.append(lexical_similarity(text, comp_text, method="cosine"))
                jaccard_scores.append(lexical_similarity(text, comp_text, method="jaccard"))
            if IoU_break:
                continue
            cosine_scores = np.array(cosine_scores)
            jaccard_scores = np.array(jaccard_scores)
            if len(cosine_scores) > 0:
                cosine_scores = softmax(cosine_scores)

            if len(jaccard_scores) > 0:
                jaccard_scores = softmax(jaccard_scores)
            combined_scores = (cosine_scores + jaccard_scores) / 2 if len(cosine_scores) > 0 else []
            
            max_prob = np.argmax(combined_scores)

            block = document.get_block(overlapping_blocks[max_prob])
            block.heading_level = level
            block.doc_toc_level = True
        
        return

    def __call__(self, document: Document):
        document.verify_headers()
        toc_info = self.get_document_toc(document.filepath)
        toc_flag = False
        if toc_info:
            self.merge_document_toc(document, toc_info)
            toc_flag = True

        toc = []
        parent_level = 0
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                if toc_flag:
                    if block.doc_toc_level:
                        level = block.heading_level
                        parent_level = level
                    else: 
                        level = max(0, parent_level + 1)
                else:
                    level = block.heading_level
                block.heading_level = level
                toc.append({
                    "title": block.raw_text(document).strip(),
                    "heading_level": level,
                    "page_id": page.page_id,
                    "bbox": block.polygon.bbox,
                    "doc_toc_level": block.doc_toc_level,
                    'line width': block.line_height(document)
                })
        document.table_of_contents = toc
