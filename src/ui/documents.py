import logging
import os
import tempfile

import streamlit as st

from ui import components
from utils.helper import (
    estimated_minutes,
    get_friendly_error_message,
    ingest_rag_documents,
)

logger = logging.getLogger(__name__)


def ingest_uploaded_pdfs(uploaded_files, thread_id: str, container):
    """Index each uploaded PDF for this conversation and return the names that succeeded.

    Progress is drawn as an indexing card inside `container` (the assistant's bubble).
    A card disappears when its file is done; a file that fails leaves a warning instead.
    """
    indexed_names = []

    for uploaded_pdf in uploaded_files:
        temporary_file_path = None
        with container:
            card = st.empty()
        card.markdown(
            components.indexing_card(uploaded_pdf.name, 0, 0, 0), unsafe_allow_html=True
        )

        def report_progress(embedded, total, name=uploaded_pdf.name, card=card):
            card.markdown(
                components.indexing_card(
                    name, embedded, total, estimated_minutes(total - embedded)
                ),
                unsafe_allow_html=True,
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
            card.empty()

        except Exception as error:
            logger.exception("Failed to index uploaded PDF %s", uploaded_pdf.name)
            card.warning(
                f"Couldn't index {uploaded_pdf.name}. {get_friendly_error_message(error)}",
                icon=":material/warning:",
            )

        finally:
            if temporary_file_path and os.path.exists(temporary_file_path):
                os.remove(temporary_file_path)

    return indexed_names
