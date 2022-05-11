# kb-model-editing

advanced nlp project (knowledge base model editing)

## Install

Run `bash scripts/install_rome.sh`. This should clone rome into the project directory,
and setup the environment correctly.
A new conda environment `rome` should become available afterwards.

## Caveat

The `rome` codebase uses a package called `dotenv` for handling environment variables.
This unfortunate design means we need to have a `.env` file at the working directory
for the code to run correctly.

The key lines in a `.env` file are `COUNTERFACT_DIR`, `TFIDF_DIR`, and `HPARAMS_DIR`.
They should point to the locations of those directories in the ROME codebase (relative to the cwd).