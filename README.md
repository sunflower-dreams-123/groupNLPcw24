# NLP Coursework 2024 — Abbreviation & Long-Form Detection

A Named Entity Recognition (NER) system that identifies **abbreviations** and their **long-form expansions** in text. Built as a group coursework project, the system fine-tunes `sentence-transformers/all-mpnet-base-v2` on the [PLOD-CW dataset](https://huggingface.co/datasets/surrey-nlp/PLOD-CW) and serves predictions via a FastAPI web interface, deployable locally or with Docker.

---

## Features

- Fine-tunes `all-mpnet-base-v2` for token classification across three entity classes: **Other**, **Abbreviation (AC)**, and **Long Form (LF)**
- Web interface for entering text and visualising per-token predictions
- Interaction logging to `predictions.log`
- Confusion matrix generation to evaluate model performance
- Docker support for containerised deployment

---

## Project Structure

```
.
├── buildmodel.py               # Fine-tunes the model on PLOD-CW and saves it
├── fastapideploy.py            # FastAPI server serving the NER web UI
├── main.py                     # Entry point / utility script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yaml         # Docker Compose configuration
├── build and run.sh            # Shell script to build and launch the container
├── templates/
│   └── ner_template.html       # Jinja2 HTML template for the web UI
├── testingendpointnotebook.ipynb  # Notebook for testing the API endpoint
└── __pycache__/
```

---


## Usage

### 1. Train the Model

Run `buildmodel.py` to download the PLOD-CW dataset, fine-tune the model, evaluate it on the test set, and save the trained model locally:

```bash
python buildmodel.py
```

This will:
- Fine-tune `sentence-transformers/all-mpnet-base-v2` for 12 epochs with early stopping
- Save the model to `./all-mpnet-base-v2-finetuned-NER/`
- Save a confusion matrix plot to `confusionmatrix.png`

### 2. Run the Web Server

Once the model is trained and saved, start the FastAPI server:

```bash
python fastapideploy.py
```

The app will be available at [http://localhost:8000](http://localhost:8000). Enter any text into the form to see per-token NER predictions (Other, Abbreviation, Long Form).

### 3. Run with Docker

Build and run the container using the provided script:

```bash
bash "build and run.sh"
```

Or with Docker Compose directly:

```bash
docker-compose up --build
```

---

## NER Labels

| Label | Description |
|-------|-------------|
| `B-O` | Other (not an entity) |
| `B-AC` | Beginning of an Abbreviation |
| `B-LF` / `I-LF` | Beginning / Inside of a Long Form |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `sentence-transformers/all-mpnet-base-v2` |
| Dataset | `surrey-nlp/PLOD-CW` |
| Epochs | 12 (with early stopping, patience=3) |
| Batch size | 4 |
| Learning rate | 6e-5 |
| Weight decay | 0.001 |
| Metric for best model | F1 |

---

## Dependencies

Key libraries used (see `requirements.txt` for full pinned versions):

- `transformers` — model loading, tokenisation, training
- `datasets` — PLOD-CW dataset loading
- `sentence-transformers` — base model
- `fastapi` + `uvicorn` — web server
- `seqeval` — NER evaluation metrics
- `scikit-learn` + `seaborn` + `matplotlib` — confusion matrix
- `torch` — model inference

---

## Dataset

This project uses the [PLOD-CW](https://huggingface.co/datasets/surrey-nlp/PLOD-CW) dataset, a benchmark for detecting abbreviations and their long-form expansions in scientific text.
