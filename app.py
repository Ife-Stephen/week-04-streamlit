
import hashlib
from io import BytesIO

import requests
import streamlit as st
from pypdf import PdfReader

from rag import build_index, retrieve, generate_answer


# 1. EXTRACT TEXT FROM PDF OR TXT

def extract_document_text(file_bytes, filename):
    """Extract readable text from PDF or TXT files."""

    if filename.lower().endswith(".pdf"):

        pdf = PdfReader(BytesIO(file_bytes))

        if pdf.is_encrypted:
            raise ValueError(
                "This PDF is password-protected. "
                "Please upload an unprotected PDF."
            )

        text = "\n\n".join(
            page.extract_text() or ""
            for page in pdf.pages
        )

        if not text.strip():
            raise ValueError(
                "No readable text was found in this PDF. "
                "It may be a scanned document. "
                "Please use a text-based PDF."
            )

        return text

    elif filename.lower().endswith(".txt"):

        try:
            text = file_bytes.decode("utf-8")

        except UnicodeDecodeError as e:
            raise ValueError(
                "Unable to read this TXT file. "
                "Please save it using UTF-8 encoding."
            ) from e

        if not text.strip():
            raise ValueError("The uploaded TXT file is empty.")

        return text

    else:
        raise ValueError("Unsupported file type. Upload a PDF or TXT file.")


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="Company Handbook RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Company Handbook RAG Assistant")

st.write(
    "Upload a PDF or TXT document and ask questions about its contents. "
    "Answers are generated using information retrieved from your document."
)


# 3. INITIALIZE SESSION STATE

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# 4. FILE UPLOADER

st.subheader("Upload Your Knowledge Base")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT document",
    type=["pdf", "txt"],
    help="Upload a PDF or plain text document to create your knowledge base."
)


# 5. PROCESS UPLOADED DOCUMENT

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # Generate a unique hash for the document
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    # Process only when a new or changed document is uploaded
    if current_hash != st.session_state.document_hash:

        # Clear the previous document index
        st.session_state.chunks = []
        st.session_state.embeddings = []
        st.session_state.document_hash = None
        st.session_state.document_name = None
        st.session_state.messages = []

        try:

            with st.spinner(
                "Reading document and generating embeddings..."
            ):

                # Extract text according to the file type
                document_text = extract_document_text(
                    file_bytes,
                    uploaded_file.name
                )

                # Create chunks and embeddings
                chunks, embeddings = build_index(document_text)

            # Store the document index in session state
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.document_hash = current_hash
            st.session_state.document_name = uploaded_file.name

            st.success(
                f"Successfully processed {uploaded_file.name}!"
            )

            st.success(
                f"Indexed {len(chunks)} text chunks."
            )

        except requests.exceptions.HTTPError as e:

            st.error(f"API HTTP error: {e}")

            if e.response is not None:
                st.error(e.response.text[:1000])

        except requests.exceptions.Timeout:

            st.error(
                "The API request timed out. "
                "Please try uploading the document again."
            )

        except requests.exceptions.RequestException as e:

            st.error(f"Network/API error: {e}")

        except (ValueError, KeyError, TypeError) as e:

            st.error(f"Document processing error: {e}")

        except Exception as e:

            st.error(f"Unexpected error while processing document: {e}")


else:

    # Clear the document index when the upload is removed
    st.session_state.chunks = []
    st.session_state.embeddings = []
    st.session_state.document_hash = None
    st.session_state.document_name = None
    st.session_state.messages = []


# 6. DISPLAY DOCUMENT INFORMATION

if st.session_state.chunks:

    st.divider()

    st.subheader("Knowledge Base Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Uploaded Document",
            st.session_state.document_name
        )

    with col2:
        st.metric(
            "Indexed Chunks",
            len(st.session_state.chunks)
        )

    st.success(
        "Your knowledge base is ready. You can now ask questions."
    )

else:

    st.info(
        "Upload a PDF or TXT document to get started."
    )


# 7. DISPLAY CHAT HISTORY

st.divider()

st.subheader("Ask Your Questions")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display retrieved source chunks
        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View retrieved source chunks"):

                for index, (score, chunk) in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source {index} — Similarity: {score:.3f}**"
                    )

                    st.write(chunk)


# 8. QUESTION INPUT

question = st.chat_input(
    "Ask a question about your document...",
    disabled=not bool(st.session_state.chunks)
)


# 9. RETRIEVE AND GENERATE ANSWER

if question:

    # Save the user's question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                # Retrieve relevant document chunks
                ranked = retrieve(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings
                )

                # Generate an answer using Groq
                response = generate_answer(
                    question,
                    ranked
                )

            st.markdown(response)

            # Display retrieved sources
            if ranked:

                with st.expander("View retrieved source chunks"):

                    for index, (score, chunk) in enumerate(
                        ranked,
                        start=1
                    ):

                        st.markdown(
                            f"**Source {index} — Similarity: {score:.3f}**"
                        )

                        st.write(chunk)

            # Save the assistant response and sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": ranked
            })

        except requests.exceptions.HTTPError as e:

            error_message = f"API HTTP error: {e}"

            if e.response is not None:
                error_message += f"\n\n{e.response.text[:1000]}"

            st.error(error_message)

        except requests.exceptions.Timeout:

            st.error("The API request timed out. Please try again.")

        except requests.exceptions.RequestException as e:

            st.error(f"Network/API error: {e}")

        except (ValueError, KeyError, TypeError) as e:

            st.error(f"Processing error: {e}")

        except Exception as e:

            st.error(f"Unexpected error: {e}")
