# Reference Assets

LEAF-1 expects these reference assets:

```bash
assets/refseq.genome.fa
assets/refseq.genome.fa.fai
assets/hg38-blacklist.v2.bed.gz
```

The FASTA is 3.1 GB and is tracked with Git LFS. After cloning, fetch the LFS
payload if Git did not download it automatically:

```bash
git lfs install
git lfs pull --include="assets/**"
```

The reference and blacklist must match the coordinate system used when the input
fragment BEDs were generated. The manuscript experiments used the
FinaleDB-distributed GRCh38 reference flavor and hg38 blacklist.

If the `.fai` file is missing, rebuild it with samtools:

```bash
samtools faidx assets/refseq.genome.fa
```
