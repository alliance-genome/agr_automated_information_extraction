# WB data type definitions (draft, for curator review)

What counts as positive for each of the 11 WormBase data types that ABC topic classifiers are trained for. They are
used as instructions for an LLM that reads a random sample of WB papers to estimate how common each data type really
is in the WB corpus (SCRUM-6482): the curated training sets over-represent papers that the old NNC classifier scored
HIGH, so their share of positives is far above the corpus's, and the classifiers trained on them over-call.

**Status: draft, not yet reviewed by WB curators.** Corrections are welcome: edit the file of the data type, or
leave a comment on the PR.

## How they were written

- for gene expression (ATP:0000041) and catalytic activity (ATP:0000061): the curation criteria and sentence
  classes of Raciti, Van Auken, Arnaboldi et al., Database 2025, baaf063 (doi:10.1093/database/baaf063), whose
  catalytic part covers protein kinase activity only and is applied here to every enzymatic activity;
- the ACKnowledge author form's help text for each data type (`acknowledge` repo,
  `src/frontend/submission_form/src/widgets/`);
- the gene expression topic tag agent of the AI curation assistant (`agr_ai_curation`,
  `packages/alliance/agents/gene_expression_topic_tag/`) and its WB rules
  (`backend/config/group_rules/agents/gene_expression/wb.yaml`);
- the notes WB curators left on papers they marked "won't curate" or validated, in the ABC (curation statuses and
  professional-curator topic tags; author answers are not used). Each data type's file lists its most frequent
  curator notes on negatives as the evidence for its exclusions.

The ATP ontology definitions ("a publication has data about X") were too general to help.

## Files

- `prompt_header.md`: the task given to the LLM;
- `common_rules.md`: rules for every data type (own new experimental results in nematodes; reviews, methods papers,
  cited results, predictions and other organisms do not count);
- `ATP_<id>_<name>.md`: one file per data type, with what counts, what does not, and the curator-note evidence.
