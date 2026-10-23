"""Atlas browser interface. Run with: python app.py.

Buttons call backend.py directly. No custom HTTP API is defined here.
"""
import logging
import os
from pathlib import Path
import requests
# Keep local browser requests out of any configured proxy.
for name in ("NO_PROXY", "no_proxy"):
    os.environ[name] = ",".join(filter(None, [os.getenv(name, ""), "localhost", "127.0.0.1", "::1"]))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
# Import only the backend functions and constant used by this interface.
# Alias inventory because the UI also has an inventory table component.
from backend import (
    MAX_BYTES,
    add_files,
    answer_question,
    clear_workspace,
    create_workspace,
    inventory as get_inventory,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("atlas")


def ui_error(exc):
    """Show the actual error so we can diagnose failed requests."""
    if isinstance(exc, ValueError):
        return gr.Error(str(exc))

    # Print the original backend traceback and DeepSeek error in the terminal.
    logger.exception("UI backend failed")

    # Display the provider's error message in the interface.
    return gr.Error(f"{type(exc).__name__}: {exc}")

def add_ui(key, company, files, basis):
    """Read selected files, upload to the session, then refresh the screen."""
    try:
        if not files:
            raise ValueError("Select files first.")
        payload = []
        for file in files:
            path = Path(file)
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Each file must be at most 5 MB.")
            payload.append((path.name, path.read_bytes()))
        message = add_files(key, company or "", payload, basis or "")
        companies, rows = get_inventory(key)
        return message, gr.update(choices=companies, value=companies[:5]), rows, [], None, None
    except Exception as exc:
        raise ui_error(exc) from None


def ask_ui(key, question, companies, source, store_type, rerank, hybrid, history):
    """Call the backend directly and display its answer and evidence."""
    try:
        result = answer_question(key, question or "", companies or [], source, store_type, rerank, hybrid)
        # History is for display only. Each backend question is standalone.
        chat = list(history or []) + [{"role": "user", "content": question},
                                      {"role": "assistant", "content": result["answer"]}]
        return "", chat, result["documents"], result["calculations"]
    except Exception as exc:
        raise ui_error(exc) from None


def reset_ui(key):
    """Remove this session’s uploaded content and reset its display."""
    try:
        clear_workspace(key)
        return "Workspace cleared.", gr.update(choices=[], value=[]), [], [], None, None, None
    except Exception as exc:
        raise ui_error(exc) from None


def clear_conversation():
    """Clear the displayed chat and evidence while keeping uploaded files."""
    return "", [], None, None

# --------------------------------------------------
# Send Atlas Report to n8n Webhook
# --------------------------------------------------

N8N_WEBHOOK_URL = "http://localhost:5678/webhook/atlas-report"


def send_email_report(recipient, companies, history):
    """
    Send the latest Atlas research answer to n8n.

    Parameters: 
        recipient: Email address entered by the user
        companies: Selected companies in Atlas
        history: Gradio chatbot conversation

    Returns:
        A status message for the Gradio interface
    """

    # Step 1: Validate recipient email
    recipient = (recipient or "").strip()

    if not recipient:
        return "Please enter a recipient email address."

    # Step 2: Check if an Atlas answer exists
    if not history:
        return "Please ask Atlas a question before sending an email."

    # Step 3: Extract the latest question and answer
    question = ""
    report = ""

    for message in reversed(history):

        if not report and message["role"] == "assistant":
            report = message["content"]

        elif report and message["role"] == "user":
            question = message["content"]
            break

    if not question or not report:
        return "No completed Atlas report is available."

    # Step 4: Prepare the data for n8n
    data = {
        "recipient": recipient,
        "company": ", ".join(companies or []),
        "question": question,
        "report": report
    }

    # Step 5: Send data to n8n
    # Step 5: Send data to n8n and handle its response
    try:
        response = requests.post(
            N8N_WEBHOOK_URL,
            json=data,
            timeout=60
        )

        # Read the JSON response from n8n
        result = response.json()

        # Handle successful email submission
        if response.ok and result.get("status") == "success":
            return "Success: " + result.get(
                "message", "Email sent successfully."
            )

        # Handle validation or other errors
        return "Error: " + result.get(
            "message", "Unable to send the report."
        )

    except requests.exceptions.Timeout:
        return "Error: n8n did not respond within 60 seconds."

    except requests.exceptions.RequestException as error:
        return f"Error: Could not connect to n8n. {error}"

    except ValueError:
        return "Error: n8n returned an invalid response."
    
CSS = """
.gradio-container {max-width:1400px !important; margin:auto;}
#hero {padding:24px; border-radius:16px; background:#172554; margin-bottom:18px;}
#hero h1 {color:white; margin:0 0 10px;}
#hero p {color:#dbeafe; margin:0;}
"""
with gr.Blocks(title="Atlas by Northstar Systems", css=CSS,
               theme=gr.themes.Soft(primary_hue="blue")) as ui:
    gr.HTML('<div id="hero"><h1>Atlas — Company Research Assistant</h1>'
            '<p>Northstar Systems’ research workspace for comparing company documents and financial tables with supporting evidence.</p></div>')
    key = gr.State("")
    ui.load(create_workspace, inputs=None, outputs=key, api_name=False)
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 1. Upload company files")
            company = gr.Textbox(label="Company label", placeholder="Apple, Microsoft or Alphabet")
            files = gr.File(label="PDF / TXT / CSV / XLSX", file_count="multiple",
                            file_types=[".pdf", ".txt", ".csv", ".xlsx"], type="filepath")
            basis = gr.Textbox(label="Reporting basis (required for tables)", lines=3,
                              placeholder="INR actual units; Jan–Mar 2026; one row per month; disjoint monthly records.")
            add = gr.Button("Add company files", variant="primary")
            status = gr.Textbox(label="Status", value="Upload one company at a time.", interactive=False)
            selected = gr.Dropdown([], multiselect=True, label="Companies to analyze", interactive=True)
            source = gr.Radio(["Documents", "Tables", "Both"], value="Both", label="Evidence source")
            with gr.Accordion("Retrieval settings (document questions)", open=True):
                store_type = gr.Radio(["FAISS", "Chroma"], value="FAISS", label="Vector store")
                rerank = gr.Checkbox(value=True, label="Rerank retrieved passages")
                hybrid = gr.Checkbox(value=False, label="Include keyword search (BM25)")
            reset = gr.Button("Clear uploaded files")
            gr.Markdown("Session files and indexes are cleared when the server restarts. "
                        "Relevant excerpts and table samples/results are sent to DeepSeek.")
        with gr.Column(scale=3, min_width=320):
            gr.Markdown("### 2. Ask and compare")
            chat = gr.Chatbot(type="messages", height=380, show_label=False)
            question = gr.Textbox(label="Question", lines=2,
                                  placeholder="Compare reported FY2024 revenue for Apple, Microsoft and Alphabet, retaining each reporting-period end.")
            with gr.Row():
                ask = gr.Button("Ask Atlas", variant="primary")
                clear = gr.Button("Clear conversation")
            
            with gr.Accordion("Uploaded files", open=True):
                inventory = gr.Dataframe(headers=["Company", "File", "Sheet/type", "Rows", "Columns", "Table ID"],
                                         interactive=False)
            with gr.Accordion("Inspect evidence", open=False):
                doc_output = gr.JSON(label="Document excerpts, sources, and reranking scores")
                calc_output = gr.JSON(label="Verified table calculations")
            gr.Markdown("Each question is standalone. State company names and the period explicitly.")
            # --------------------------------------------------
            # Email Report using n8n
            # --------------------------------------------------

            gr.Markdown("### 3. Email Research Report")

            recipient_email = gr.Textbox(
                label="Recipient Email",
                placeholder="Enter the recipient's email address"
            )

            email_button = gr.Button(
                "Email Report",
                variant="secondary"
            )

            email_status = gr.Textbox(
                label="Email Status",
                interactive=False
            )   
    # Gradio 5: keep events available to this UI, without a client-facing API.
    options = {"concurrency_limit": 1, "concurrency_id": "workspace", "api_name": False}
    add.click(add_ui, [key, company, files, basis],
              [status, selected, inventory, chat, doc_output, calc_output], **options)
    for trigger in (ask.click, question.submit):
        trigger(ask_ui, [key, question, selected, source, store_type, rerank, hybrid, chat],
                [question, chat, doc_output, calc_output], **options)
    clear.click(clear_conversation, None, [question, chat, doc_output, calc_output], **options)
    reset.click(reset_ui, [key], [status, selected, inventory, chat, doc_output, calc_output, files], **options)
    # --------------------------------------------------
    # Connect Email Report button to n8n
    # --------------------------------------------------

    email_button.click(
        fn=send_email_report,

        # Values passed to the Python function
        inputs=[
            recipient_email,
            selected,
            chat
        ],

        # Display the result in the Email Status field
        outputs=email_status,

        **options
    )
if __name__ == "__main__":
    # Gradio starts the local browser interface; no separate server setup is needed.
    ui.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False,
    )
