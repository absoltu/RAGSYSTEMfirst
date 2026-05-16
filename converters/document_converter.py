import torch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,PdfFormatOption
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)

from config import (
    USE_CUDA,
    ENABLE_OCR,
    ENABLE_TABLE_STRUCTURE
)


class DocumentParser:

    def __init__(self):

        # =====================================
        # DEVICE
        # =====================================

        if (
            USE_CUDA
            and torch.cuda.is_available()
        ):

            self.device = "cuda"

        else:

            self.device = "cpu"

        print(
            f"Using device: {self.device}"
        )

        # =====================================
        # PDF PIPELINE OPTIONS
        # =====================================

        pipeline_options = PdfPipelineOptions()

        # OCR
        pipeline_options.do_ocr = ENABLE_OCR

        # TABLES
        pipeline_options.do_table_structure = (
            ENABLE_TABLE_STRUCTURE
        )

        # FORMULAS - Enable formula recognition
        pipeline_options.do_formula_enrichment = False

        # =====================================
        # CREATE CONVERTER
        # =====================================

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    # =====================================
    # PARSE DOCUMENT
    # =====================================

    def parse_document(
        self,
        file_path: str
    ):

        result = self.converter.convert(
            file_path
        )

        return result.document

    # =====================================
    # EXPORT DOCUMENT
    # =====================================

    def export_document(
        self,
        document,
        output_format: str = "markdown"
    ):

        """
        output_format:
            - markdown
            - text
            - json
        """

        if output_format == "markdown":

            return (
                document.export_to_markdown()
            )

        elif output_format == "text":

            return (
                document.export_to_text()
            )

        elif output_format == "json":

            return (
                document.export_to_dict()
            )

        else:

            raise ValueError(
                f"Unsupported format: "
                f"{output_format}"
            )

    # =====================================
    # EXTRACT STRUCTURE WITH SECTIONS
    # =====================================

    def extract_sections(
        self,
        document
    ):
        """
        Extract text with section headers.
        Returns list of tuples: (text, section_title, section_level)
        """
        sections = []
        current_section = "Document"
        current_level = 0

        for item in document.body:
            # Check if it's a heading
            if hasattr(item, 'tag') and 'heading' in item.tag.lower():
                # Extract heading text
                if hasattr(item, 'export_to_markdown'):
                    current_section = item.export_to_markdown().strip().lstrip('#').strip()
                elif hasattr(item, 'text'):
                    current_section = item.text
                current_level = int(item.tag[-1]) if item.tag[-1].isdigit() else 1
            
            # For text items, include section info
            elif hasattr(item, 'export_to_markdown'):
                text = item.export_to_markdown().strip()
                if text:
                    sections.append({
                        "text": text,
                        "section": current_section,
                        "level": current_level
                    })
            elif hasattr(item, 'text'):
                text = item.text.strip() if hasattr(item.text, 'strip') else str(item.text)
                if text:
                    sections.append({
                        "text": text,
                        "section": current_section,
                        "level": current_level
                    })

        return sections

    # =====================================
    # FULL CONVERT
    # =====================================

    def convert(
        self,
        file_path: str,
        output_format: str = "markdown"
    ):

        document = self.parse_document(
            file_path
        )

        return self.export_document(
            document,
            output_format
        )