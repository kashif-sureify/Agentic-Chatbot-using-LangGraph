import logging
import os
import tempfile

import streamlit as st

from utils.helper import estimated_minutes, ingest_rag_documents

logger = logging.getLogger(__name__)


def ingest_uploaded_pdfs(uploaded_files, thread_id: str):
    """Index each uploaded PDF for this conversation and return the names that succeeded."""
    indexed_names = []

    for uploaded_pdf in uploaded_files:
        temporary_file_path = None
        progress_bar = st.progress(0.0, text=f"Reading {uploaded_pdf.name} …")

        def report_progress(embedded, total, name=uploaded_pdf.name):
            minutes = estimated_minutes(total - embedded)
            remaining = f" — about {minutes} min left" if minutes else ""
            progress_bar.progress(
                embedded / total if total else 1.0,
                text=f"Embedding {name}: {embedded} of {total} chunks{remaining}",
            )

        try:
            # Closed before loading: PyMuPDF reads from disk, and a PDF smaller than
            # Python's write buffer would still be an empty file at this point
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_pdf.getvalue())
                temporary_file_path = temp_file.name

            ingest_rag_documents(
                temporary_file_path,
                thread_id,
                source_name=uploaded_pdf.name,
                on_progress=report_progress,
            )

            indexed_names.append(uploaded_pdf.name)

        except Exception as error:
            logger.exception("Failed to index uploaded PDF %s", uploaded_pdf.name)
            st.error(f"Could not read {uploaded_pdf.name}: {error}")

        finally:
            progress_bar.empty()
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

    return indexed_names
