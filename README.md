# Citation prediction task

Given a citation context, predict the most likely citation.

Given a citation context, retrieve a ranked list of k related papers.

## Data

Download the xml files from: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/

## baseline systems

## elastic search retrieval

First step: insert the abstract of every paper into the 

Index the papers with ES. Retrieve top k.

Indices: abstract, full text, aggregate_context

How to control aggregate_contexts based on cutoff date?

## Evaluation

nDCG
MAP

## Re-ranking

## baseline data preparation

Extract citation sentences.
Remove the citation anchors and author names.

Separate based on a date cutoff

## TODO 1/8/2021

0. Organize module structure (DONE)
1. Flake8 (DONE)
2. Adjust loading for module structure
3. Load model in eval script
4. Implement re-ranker with model output
5. Run evaluation with re-ranking

model, optimizers = amp.initialize(model, optimizers, opt_level='02')

with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
