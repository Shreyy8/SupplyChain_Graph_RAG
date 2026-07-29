# graph_rag/ingestion/pdf_loader.py
#
# PURPOSE:
#   Reads every PDF file from the data/pdfs/ folder and extracts the raw
#   text from each page. Returns a list of LangChain Document objects —
#   one Document per page — each containing the page text and metadata
#   about which file and page number it came from.
#
#   This is the very first step in the pipeline:
#   PDF files --> raw text --> (next step: text_chunker.py splits into chunks)
#
# DO YOU RUN THIS FILE?
#   No. This module is imported and called by scripts/run_ingestion.py.
#   You will never run this file directly.
#
# WHAT IS A LANGCHAIN DOCUMENT?
#   A Document is a simple container with two fields:
#     - page_content : the raw text string extracted from the page
#     - metadata     : a dictionary with extra info like source filename,
#                      page number, and total page count
#   LangChain uses Documents as the standard data format throughout the
#   pipeline, so we produce them here and pass them downstream.


import pdfplumber                          # extracts clean text from PDF files
from pathlib import Path                   # handles file paths cleanly across Mac/Windows/Linux
from typing import List                    # type hint for list return values
     # LangChain's standard text container
from loguru import logger                  # structured logging with timestamps and colours
from langchain_core.documents import Document

def load_pdfs_from_directory(pdf_dir: str | Path) -> List[Document]:
    """
    Reads all PDF files from the given directory and extracts text page by page.

    For each PDF file found, we open it with pdfplumber and iterate through
    every page. Each page's text becomes one LangChain Document object.
    We attach metadata so that later in the pipeline we always know which
    document and page a piece of text came from.

    Args:
        pdf_dir: Path to the folder containing PDF files.
                 Can be a string like "./data/pdfs" or a Path object.
                 All .pdf files in this folder will be processed.

    Returns:
        A list of LangChain Document objects, one per page across all PDFs.
        Example: 5 PDFs with an average of 4 pages each = ~20 Documents.

    Raises:
        FileNotFoundError: if the pdf_dir folder does not exist.
        ValueError: if no PDF files are found in the folder.
    """

    # Convert the input to a Path object so we can use .glob(), .exists(), etc.
    # This works whether the caller passes a string or a Path
    pdf_dir = Path(pdf_dir)

    # Check that the directory actually exists before trying to read from it
    # Fail early with a clear message rather than a confusing pdfplumber error
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_dir.resolve()}\n"
            f"Create it with:  mkdir -p {pdf_dir}\n"
            f"Then place your PDF files inside it."
        )

    # Find all .pdf files in the directory (non-recursive — we only look in
    # the top level, not in subdirectories)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    # If no PDFs were found, raise a clear error rather than returning
    # an empty list silently — an empty list would cause confusing errors later
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {pdf_dir.resolve()}\n"
            f"Make sure your five PDF files are placed in that folder."
        )

    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    # This list will hold all Document objects produced from all PDFs
    all_documents: List[Document] = []

    # Process each PDF file one at a time
    for pdf_path in pdf_files:

        logger.info(f"Loading: {pdf_path.name}")

        try:
            # Open the PDF with pdfplumber
            # pdfplumber handles the low-level PDF parsing and gives us
            # clean text for each page — much better than PyPDF2 for
            # documents with tables and mixed layouts like our financial reports
            with pdfplumber.open(pdf_path) as pdf:

                # Record total pages for metadata — useful for debugging
                total_pages = len(pdf.pages)

                # Iterate through every page in this PDF
                for page_number, page in enumerate(pdf.pages, start=1):

                    # Extract the text from this page as a plain string
                    # extract_text() returns None if the page has no extractable
                    # text (e.g. a scanned image page) — we handle that below
                    page_text = page.extract_text()

                    # Skip pages with no extractable text
                    # This can happen with blank pages or image-only pages
                    if not page_text or not page_text.strip():
                        logger.debug(
                            f"Skipping page {page_number} of {pdf_path.name} "
                            f"— no extractable text found"
                        )
                        continue

                    # Clean the extracted text slightly:
                    # strip() removes leading and trailing whitespace
                    # This prevents empty strings or whitespace-only pages
                    # from making it into the pipeline
                    clean_text = page_text.strip()

                    # Build the metadata dictionary for this page
                    # This metadata travels with the text through the entire
                    # pipeline and ends up stored in ChromaDB alongside the vector
                    # so that we always know where a retrieved chunk came from
                    metadata = {
                        # The filename without the directory path
                        # e.g. "apple_annual_overview_FY2024.pdf"
                        "source": pdf_path.name,

                        # The full absolute path — useful for debugging
                        "file_path": str(pdf_path.resolve()),

                        # Which page this text came from (1-indexed, human-friendly)
                        "page": page_number,

                        # Total pages in the source document
                        # Useful context when reading retrieved chunks
                        "total_pages": total_pages,

                        # The company name — derived from the filename
                        # e.g. "apple_annual_overview_FY2024.pdf" -> "apple"
                        # We take the first word before the first underscore
                        "company": pdf_path.stem.split("_")[0].upper(),
                    }

                    # Create a LangChain Document object for this page
                    # page_content holds the text; metadata holds the context info
                    document = Document(
                        page_content=clean_text,
                        metadata=metadata,
                    )

                    # Add this page's Document to our running list
                    all_documents.append(document)

            logger.success(
                f"Loaded {pdf_path.name} — "
                f"{total_pages} pages, "
                f"{sum(1 for d in all_documents if d.metadata['source'] == pdf_path.name)} "
                f"pages with extractable text"
            )

        except Exception as e:
            # Log the error but continue processing remaining PDFs
            # We do not want one bad PDF to stop the entire ingestion run
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue

    # Final summary log so the operator can see the ingestion result at a glance
    logger.info(
        f"PDF loading complete — "
        f"{len(pdf_files)} files processed, "
        f"{len(all_documents)} total pages extracted"
    )

    return all_documents


def get_pdf_summary(documents: List[Document]) -> dict:
    """
    Produces a brief summary of the loaded documents grouped by source file.

    Useful for logging and for verifying that all five PDFs were loaded
    correctly before moving to the chunking step.

    Args:
        documents: The list of Document objects returned by load_pdfs_from_directory().

    Returns:
        A dictionary where each key is a source filename and each value is
        the number of pages loaded from that file.
        Example:
            {
                "apple_annual_overview_FY2024.pdf": 4,
                "tsmc_manufacturing_report_FY2023.pdf": 5,
                ...
            }
    """

    # Count how many Document objects came from each source file
    summary = {}
    for doc in documents:
        # Get the source filename from this document's metadata
        source = doc.metadata.get("source", "unknown")

        # Increment the page count for this source
        # dict.get(key, 0) returns 0 if the key does not exist yet
        summary[source] = summary.get(source, 0) + 1

    return summary