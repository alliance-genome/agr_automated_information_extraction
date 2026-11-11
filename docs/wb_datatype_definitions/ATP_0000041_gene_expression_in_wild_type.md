# ATP:0000041 gene expression in wild type

**Status:** draft, not yet reviewed by WB curators.

Positive: the paper reports its own low-throughput experiments that show where and/or when a specific C. elegans gene
or gene product is expressed in a WILD-TYPE background, so that a curator can name the gene and its spatial or
temporal localization: anatomy (cells, tissues), subcellular localization, and/or life stage. Methods: reporter gene
analysis (GFP etc.), antibody staining, in situ hybridization (ISH), single-molecule FISH, hybridization chain
reaction (HCR), RT-PCR, qPCR, northern blot, western blot. Explicitly stated ABSENCE of expression in a tissue or life
stage also counts. Examples: "A transcriptional reporter for nlp-18 revealed strong expression in the intestine and
three pairs of neurons"; "gmap-1 and gmap-2/3 genes are both expressed in epithelial cells". The details are often
spread over several adjoining sentences (a gene in one, its tissues or stage in the next); together they count.
Not positive:
- changes of expression or localization in mutants, RNAi, or after a treatment, stress or infection: that is gene
  regulation (ATP:0000070), not wild-type expression;
- expression constructs used for rescue, tissue-specific or ectopic expression, or other genetic manipulation
  (e.g. "Expression of CKR-1 in AIB rescued the ckr-1 mutant defects"), and reporters or promoters used only as
  markers to label cells or structures (e.g. a neuron marker, a microtubule marker), not to describe that gene's own
  expression;
- only the experimental set-up, a hypothesis or rationale ("we hypothesized that NMGP-1 is expressed in neurons"), or
  methods ("gonads were prepared for RNA ISH") without the expression result;
- high-throughput expression profiling only: microarrays, RNA-seq, SAGE, proteomics, without gene-specific anatomy or
  stage results;
- expression of C. elegans genes in heterologous systems (Xenopus oocytes, CHO or other cell lines);
- expression only in other species.

## Evidence: source of the definition

Positive and negative definitions follow Raciti, Van Auken, Arnaboldi et al., "Characterization and automated
classification of sentences in the biomedical literature: a case study for biocuration of gene expression and protein
kinase activity", Database 2025, baaf063 (doi:10.1093/database/baaf063, PMID 41026497): its gene expression curation
criteria and its fully curatable / related language sentence classes (examples from its Table 1).

## Evidence: most frequent WB curator notes on negatives

From WB curation statuses ("won't curate") and professional-curator topic tags in the ABC (prod, 2026-11-06). Notes that only record the triage process ("SVM positive, curation negative", "pre-2001") are left out.

| Papers | Curator note |
|---|---|
| 84 | gene regulation |
| 11 | mouse |
| 10 | microarrays |
| 8 | rnaseq |
| 8 | microarray |
| 7 | genereg |
| 6 | drosophila |
| 6 | non elegans |
| 5 | methods |
| 5 | large scale |
| 4 | review |
| 4 | rat |
| 3 | book chapter |
| 3 | sage |
| 3 | fly |
