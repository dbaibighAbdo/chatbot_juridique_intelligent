
# KG Construction

**Description:** This folder contains utilities to load PDF legal documents, preprocess them and populate a Neo4j knowledge graph using `neo4j-graphrag` components. The main example script is `data_loader_pdf.py` which demonstrates a PDF loader (with small custom cleaning) and a pipeline that builds/embeds content into Neo4j.

**Requirements:**
- **Python:** `>=3.12` (as declared in the project `pyproject.toml`).
- **Key packages:** `neo4j==5.28.2`, `neo4j-graphrag==1.10.1`, `fsspec==2024.12.0`, and a dotenv loader for environment variables. These are listed in the project manifest.

**Environment variables:**
Create a `.env` file (in the repository root or working dir) with these variables filled:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j
OPENAI_API_KEY=sk-...   # required by OpenAILLM/OpenAIEmbeddings
```

**Installation (recommended):**
- From the repository root, install the project or the required packages. Example (PowerShell):

```
python -m pip install -e .
# or install the main deps explicitly:
python -m pip install "neo4j==5.28.2" "neo4j-graphrag==1.10.1" "fsspec==2024.12.0" python-dotenv
```

**Usage:**
- Place PDF files inside the `kg_construction/data/` folder (the example script expects `data/مدونة الشغل المغربية.pdf`).
- Edit `kg_construction/data_loader_pdf.py` to change the `pdf_file` variable or point it to the PDF you want to process.
- Run the loader from the repository root (PowerShell):

```
python kg_construction/data_loader_pdf.py
```

What the script does:
- Loads environment variables with `dotenv`.
- Connects to Neo4j using the `NEO4J_*` settings.
- Configures an OpenAI-based LLM and embeddings (the script references `OpenAILLM` and `OpenAIEmbeddings`). Make sure `OPENAI_API_KEY` is set.
- Uses a `CustomPDFLoader` (small custom cleaning via regex) to load and clean PDF text.
- Runs `SimpleKGPipeline` to build/insert data into Neo4j and produce embeddings.

**Notes & troubleshooting:**
- Ensure Neo4j is running and reachable at `NEO4J_URI` and that the username/password are correct.
- The `neo4j-graphrag` package internally may require OpenAI credentials; set `OPENAI_API_KEY` in the `.env` or environment.
- If you see Unicode or path issues on Windows, ensure the file path is correct and Python can open the PDF (use raw strings or escape backslashes).
- The script currently hardcodes `pdf_file` — you can modify the script to accept a command-line argument for convenience.

