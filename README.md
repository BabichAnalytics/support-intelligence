# Support Intelligence

A local AI support-ticket analytics platform with Python, SQLite, an interactive dashboard, and a trained **multinomial Naive Bayes classifier**. No paid APIs, remote inference, or third-party Python dependencies.

## Start

Install Python 3.10+, download the repository, then run in its folder:

```bash
python app.py
```

Open **http://127.0.0.1:8002** in your browser. Keep the terminal open. Ctrl+C stops the server. Use `python3` if needed.

## What works

- CSV validation and transactional imports to SQLite
- Idempotent imports: existing ticket IDs update rather than duplicate
- Category prediction learned locally from `training.csv`
- Separate, explicit high-priority keyword rules
- Search and category/status filters
- Category counts and recurring subject-word analysis
- Per-ticket review and clearly marked template reply drafts
- Synthetic labeled-data evaluation, plus tests

The classifier is genuine local statistical machine learning, **not a generative language model**. Scores are relative model scores, not calibrated confidence. Unseen vocabulary yields “Needs review.” Reply drafts are templates, not AI-generated responses. No replies are sent, and review does not persist edits in this milestone.

## Data and evaluation

`training.csv` contains fictional labeled examples. `sample_tickets.csv` contains 150 fictional tickets with optional expected labels. Training and sample texts differ, but share curated vocabulary. Demo accuracy does not establish real-world performance; validate using independently labeled, representative tickets before operational use. High priority is rules-based and is not an incident-response guarantee.

Required import fields: `id,created,subject,description,status`. Dates use YYYY-MM-DD. Status is Open, In progress, or Resolved. Optional `expected_category` is Access, Network, Hardware, Software, or Billing.

```bash
python app.py --csv your_tickets.csv --db your_support.db
python app.py --import-only
python -m unittest -v
```

Imports update matching IDs; existing unrelated records remain. Use a separate database to isolate datasets. Dashboard totals reflect filters; evaluation accuracy covers all labeled records in the database. The queue displays the first 200 matching tickets.

SQLite data stays local. The server binds only to 127.0.0.1 and provides read-only HTTP endpoints. It is a local portfolio demonstration, not a hardened hosted service. Do not commit real support-ticket data, credentials, or local database files.

## Next milestones

1. Persist human corrections and measure classification errors by category.
2. Add an optional local language model for draft generation after checking hardware and current model setup instructions.
3. Add a proper held-out real-world evaluation, ingestion audit log, and packaging.

This milestone is runnable now; the roadmap features are not implemented yet.
