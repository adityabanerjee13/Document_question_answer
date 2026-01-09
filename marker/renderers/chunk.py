import html
from typing import List, Dict, Optional, Annotated, Tuple

from bs4 import BeautifulSoup
from pydantic import BaseModel

from marker.renderers.json import JSONRenderer, JSONBlockOutput
from marker.schema.document import Document
from marker.settings import settings
from marker.renderers.markdown import cleanup_text, Markdownify


class FlatBlockOutput(BaseModel):
    id: str
    block_type: str
    html: str
    page: int
    polygon: List[List[float]]
    bbox: List[float]
    section_hierarchy: Dict[int, str] | None = None
    images: dict | None = None


class ChunkOutput(BaseModel):
    blocks: List[FlatBlockOutput]
    page_info: Dict[int, dict]
    metadata: dict

def collect_images(block: JSONBlockOutput) -> dict[str, str]:
    if not getattr(block, "children", None):
        return block.images or {}
    else:
        images = block.images or {}
        for child_block in block.children:
            images.update(collect_images(child_block))
        return images

def assemble_html_with_images(block: JSONBlockOutput, image_blocks: set[str]) -> str:
    if not getattr(block, "children", None):
        block_path = (block.id).replace('/', '_')
        image_name = f"{block_path}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
        if block.block_type in image_blocks:
            return f"<p>{block.html}<img src='{image_name}'></p>"
        else:
            return block.html

    child_html = [assemble_html_with_images(child, image_blocks) for child in block.children]
    child_ids = [child.id for child in block.children]

    soup = BeautifulSoup(block.html, "html.parser")
    content_refs = soup.find_all("content-ref")
    for ref in content_refs:
        src_id = ref.attrs["src"]
        if src_id in child_ids:
            ref.replace_with(child_html[child_ids.index(src_id)])

    return html.unescape(str(soup))

def json_to_chunks(
    block: JSONBlockOutput, image_blocks: set[str], page_id: int=0) -> FlatBlockOutput | List[FlatBlockOutput]:
    if block.block_type == "Page":
        children = block.children
        page_id = int(block.id.split("/")[-1])
        return [json_to_chunks(child, image_blocks, page_id=page_id) for child in children]
    else:
        return FlatBlockOutput(
            id=block.id,
            block_type=block.block_type,
            html=assemble_html_with_images(block, image_blocks),
            page=page_id,
            polygon=block.polygon,
            bbox=block.bbox,
            section_hierarchy=block.section_hierarchy,
            images=collect_images(block),
        )


class ChunkRenderer(JSONRenderer):
    page_separator: Annotated[
        str, "The separator to use between pages.", "Default is '-' * 48."
    ] = "-" * 48
    inline_math_delimiters: Annotated[
        Tuple[str], "The delimiters to use for inline math."
    ] = ("$", "$")
    block_math_delimiters: Annotated[
        Tuple[str], "The delimiters to use for block math."
    ] = ("$$", "$$")
    paginate_output: Annotated[
        bool,
        "Whether to paginate the output.",
    ] = False


    @property
    def md_cls(self):
        return Markdownify(
            self.paginate_output,
            self.page_separator,
            heading_style="ATX",
            bullets="-",
            escape_misc=False,
            escape_underscores=True,
            escape_asterisks=True,
            escape_dollars=True,
            sub_symbol="<sub>",
            sup_symbol="<sup>",
            inline_math_delimiters=self.inline_math_delimiters,
            block_math_delimiters=self.block_math_delimiters,
        )

    def __call__(self, document: Document) -> Optional[ChunkOutput|Dict[str, dict]]:
        document_output = document.render(self.block_config)
        json_output = []
        for page_output in document_output.children:
            json_output.append(self.extract_json(document, page_output))

        # This will get the top-level blocks from every page
        chunk_output = []
        for item in json_output:
            chunks = json_to_chunks(item, set([str(block) for block in self.image_blocks]))
            chunk_output.extend(chunks)
        page_info = {
            page.page_id: {"bbox": page.polygon.bbox, "polygon": page.polygon.polygon}
            for page in document.pages
        }

        if self.output_json:
            out = {
                "blocks": [chunk.model_dump() for chunk in chunk_output],
                "page_info": page_info,
            }
            temp = {}
            temp_imgs = {}

            for block in out['blocks']:
                temp_id = str(block['id']).split('/')
                chunk_html = block['html']
                markdown = self.md_cls.convert(chunk_html)
                markdown = cleanup_text(markdown)

                if markdown.strip() and temp_id[-2] in ['Text', 'SectionHeader', 'ListGroup']:  # Only include blocks with non-empty markdown
                    temp[str(block['id'])] = {
                        'page': int(temp_id[2]),
                        'block_id': int(temp_id[-1]),
                        'block_type': temp_id[-2],
                        'html': chunk_html,
                        'markdown': markdown,
                        'bbox': block['bbox'],
                    }
                    temp_imgs.update(block['images'] if block['images'] else {})
                elif not (temp_id[-2] in ['Text', 'SectionHeader', 'ListGroup']):
                    temp[str(block['id'])] = {
                        'page': int(temp_id[2]),
                        'block_id': int(temp_id[-1]),
                        'block_type': temp_id[-2],
                        'html': chunk_html,
                        'markdown': markdown,
                        'bbox': block['bbox'],
                    }
                    temp_imgs.update(block['images'] if block['images'] else {})
            return temp, temp_imgs, self.generate_document_metadata(document, document_output)
        
        else:
            return ChunkOutput(
                blocks=chunk_output,
                page_info=page_info,
                metadata=self.generate_document_metadata(document, document_output),
            )
