# ATP:0000061 catalytic activity

**Status:** draft, not yet reviewed by WB curators.

Positive: the paper reports its own biochemical experiments that show the enzymatic activity of a named C. elegans
gene product (e.g. kinase, phosphatase, protease, methyltransferase, ATPase/ATP hydrolysis, ligase, deacetylase),
typically an in vitro assay of the purified, recombinant or immunoprecipitated protein, so that the result could
support a GO molecular function annotation for that activity. What is needed is the enzyme, the activity and the
assay; the substrate is not required (when given, it is a bonus). Examples: "When this activated PMK-1 was tested in
the in vitro kinase assay, it was found to efficiently phosphorylate GST-SKN-1"; "AIR-2 phosphorylates REC-8 in
vitro"; summary statements of the paper's own results such as "UNC-43 is a calcium-dependent kinase capable of
phosphorylating CaMKII substrates". The details are often spread over several adjoining sentences (the assay in one,
the enzyme or the result in the next); together they count.
Not positive:
- genetic or phenotypic evidence only (mutant phenotypes of an enzyme gene, "kinase-independent" functions, IMP-style
  inference) with no activity assay;
- binding assays without catalysis (actin or microtubule binding, bundling, protein-protein binding, Y2H, pull-downs,
  gel shifts) - those are physical interactions;
- only the experimental set-up, a hypothesis or rationale ("we tested whether MBK-2 can phosphorylate OMA-1 in
  vitro") or methods ("catalytic activity was quantified by 32P incorporation") without the result; open questions in
  the discussion ("whether CDK-1 targets WRM-1 remains to be determined");
- enzymatic activities known from earlier papers ("phosphorylation by GSK-3 requires a priming kinase"), discussion
  of enzymatic pathways or enzyme families without assays of specific worm gene products;
- assays of bacterial or other non-worm enzymes; metabolite quantification not tied to a specific gene product.

## Evidence: source of the definition

Follows the protein kinase activity criteria of Raciti, Van Auken, Arnaboldi et al., "Characterization and automated
classification of sentences in the biomedical literature: a case study for biocuration of gene expression and protein
kinase activity", Database 2025, baaf063 (doi:10.1093/database/baaf063, PMID 41026497), applied to every catalytic
activity: fully curatable sentences name the enzyme, the activity and the assay, the substrate is not required for a
GO MF annotation, and the related language classes (set-up, hypothesis, previously published, discussion) are not
positive. Examples from its Table 1. The paper covers kinases only; extending its criteria to the other enzymes is
ours.

## Evidence: most frequent WB curator notes on negatives

From WB curation statuses ("won't curate") and professional-curator topic tags in the ABC (prod, 2026-11-06). Notes that only record the triage process ("SVM positive, curation negative", "pre-2001") are left out.

| Papers | Curator note |
|---|---|
| 2 | physical interactions |
| 2 | imp evidence for methyltransferase activity |
| 2 | physical interactions. |
| 1 | genetic, phenotypic, and expression analyses of svh-1 ligand/protease. |
| 1 | lots of physical interactions. |
| 1 | y2h and pull down assays for let-99 and par-5 |
| 1 | this paper has assays for in vitro actin binding and depolymerization assays, but not enzymatic activities per se. |
| 1 | genetic and phenotypic analyses of metalloproteinases. |
| 1 | other in vitro experiments - microtubule binding assay and y2h |
| 1 | microtubule binding and bundling of spd-1. |
| 1 | expressing wt and mutant human sod1 in c. elegans to study its aggregation and resulting phenotypic effects. |
| 1 | cgc5933 large set of glycosylated proteins 2003-07-23 16:40:44.120962-07 pmid12754521 large-scale glycoprotein analysis 2003-08-01 18:24:12.553234-07 |
| 1 | language positive? the analysis using phos-tag gels showed that ce_yap-1- gfp was dephosphorylated after heat shock in wild-type worms and mob-4 defic |
| 1 | secretory polarity of ins-35 and ins-7. |
| 1 | the 53bp1 homolog in c. elegans influences dna repair and promotes apoptosis in response to ionizing radiation |
