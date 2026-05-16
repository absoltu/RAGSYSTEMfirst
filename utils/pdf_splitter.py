import fitz
import os


def split_pdf(
    pdf_path: str,
    output_dir: str,
    pages_per_chunk: int = 20
):

    os.makedirs(output_dir, exist_ok=True)

    pdf = fitz.open(pdf_path)

    total_pages = len(pdf)

    split_files = []

    for start in range(
        0,
        total_pages,
        pages_per_chunk
    ):

        end = min(
            start + pages_per_chunk,
            total_pages
        )

        new_pdf = fitz.open()

        new_pdf.insert_pdf(
            pdf,
            from_page=start,
            to_page=end - 1
        )

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(
            output_dir,
            f"{base_name}_part_{start}_{end}.pdf"
        )

        new_pdf.save(output_path)

        new_pdf.close()

        split_files.append(output_path)

    pdf.close()

    return split_files
