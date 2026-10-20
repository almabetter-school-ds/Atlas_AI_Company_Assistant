# Atlas — Company Research Assistant

Northstar Systems uses Atlas to research uploaded company documents and tables.
The two-file Gradio application retains document retrieval, spreadsheet
calculations, source evidence, FAISS/Chroma, BM25 and optional reranking.

## Setup

Extract this folder and open it in VS Code. Use Python 3.11/3.12, 64-bit.

Windows PowerShell:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

macOS/Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

**Before asking a question:** edit the bundled `.env` and replace
`DEEPSEEK_API_KEY=replace_with_your_key` with your own key. Restart after edits.
If cloning from Git, copy `.env.example` to `.env` first. Never commit a real key.
Open http://127.0.0.1:7860/ and keep the terminal running.

## Read alongside the code

- `app.py`: explicit backend imports, upload/question/clear callbacks and UI.
- `backend.py`: latest commented backend; workspace management, file parsing,
  retrieval, planning and validated pandas calculations.
- `Atlas_AI_Company_Assistant.ipynb`: concise guide,
  function walkthrough, infographics, sample reporting notes and an AI build prompt.
- `sample_data/`: six sample files.
- `.env.example`: safe configuration template; `.env` is ignored by Git.

Upload each company's TXT and CSV/XLSX together. Paste its reporting basis from
the notebook. Choose Documents, Tables or Both and inspect the evidence panels.

## DeepSeek configuration

Both chains use `ChatDeepSeek`, default model `deepseek-flash`, endpoint
`https://api.deepseek.com`, with thinking disabled. Change the model in `.env`
if needed for your account. 


## Behavior and limits

The interface calls Python functions directly; no custom API service is created.
Gradio still runs its internal web server. All events have `api_name=False` and
launch uses `show_api=False` with Gradio 5.50.0. DeepSeek calls require internet
and account access. Relevant excerpts and table samples/results go to DeepSeek.
Files/indexes are in RAM, lost on restart. Chat history is display-only.
Tables support validated aggregations/filters, not arbitrary generated code,
joins or forecasting. Scanned PDFs need OCR separately. 

