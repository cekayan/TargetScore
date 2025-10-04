# TargetScore

This repository contains the TargetScore project. It was copied from a local `SENDING` folder and includes scripts, data, and utilities for processing genomic and drug response data.

What is included

- Data files and pickles (e.g., `exp_data_v1.pkl`, `genomics_data.pkl`).
- Scripts and modules for data loading, imputation, model training, and analysis (see `ModularMM3` and `ModulerMI` directories).
- Per-cell-line mean-filled response CSVs in `mean-filled-resps/`.

Assumptions and recommended environment

- This README was generated automatically. I inferred common Python packages used by the project; please review `requirements.txt` and adjust versions as needed.
- Recommended Python: 3.8+ (project may work on newer versions).

Quickstart

1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate  # on Windows (bash) use: .venv\\Scripts\\activate

2. Install dependencies:

   pip install -r requirements.txt

3. Run an example entrypoint (one of the modules under `ModulerMI` or `ModularMM3`):

   python ModulerMI/main.py

Notes about large data

- The repository contains CSV and pickle files that may be large; consider using Git LFS for large binaries if you plan to push them to GitHub.

Setting up the GitHub remote and pushing

- Create a new GitHub repository named `TargetScore` under your account (via web UI or GitHub CLI).
- Then run these commands in the `TargetScore` folder:

   git remote add origin git@github.com:<your-username>/TargetScore.git
   git branch -M main
   git push -u origin main

If you prefer HTTPS:

   git remote add origin https://github.com/<your-username>/TargetScore.git
   git push -u origin main

License

- No license was specified. Add a `LICENSE` file if you want to set an open-source license.

Contact

- If you need help preparing a lighter-weight or packaged version of this project (e.g., removing large data files, adding examples), tell me what you'd like and I'll help.