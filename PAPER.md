# Serial position predicts lexicalization: prefixed words drift from their parts more than suffixed words, in human text and in language models

**Kevin Edey**

*Draft, June 2026. Code and data: https://github.com/GenericJam/affix-novelty*

## Abstract

When a word is built from an affix and a root, its meaning is sometimes
predictable from the parts (`rerun` is "run again") and sometimes not
(`remember` is no longer "re-member"). We ask whether this loss of
compositionality is governed by the serial position of the affix rather than its
grammatical category. We hypothesize that the first morpheme a reader must
traverse to reach a word becomes its retrieval anchor, so that an affix-first
(prefixed) word binds holistically and is free to drift, whereas a root-first
(suffixed) word stays tethered to the still-meaningful root. Using validated
morphological segmentation and a vector-offset measure of compositionality, we
find that English prefixed words are non-compositional far more often than
suffixed words, across embeddings, across the frequency range, and at the level
of individual bound-root families. The asymmetry is stable across the 20th
century, replicates in direction across four Romance languages, and, most
notably, appears in the internal representations of an autoregressive language
model (GPT-2) once a standard anisotropy correction is applied. Because a
next-token predictor has neither memory nor a brain, its reproduction of the
effect implicates left-to-right processing order itself as the mechanism. We
discuss what this suggests for lexicalization as a process in language evolution.

## 1. Introduction

Lexicalization, the hardening of a composed form into a single stored unit with
its own meaning, is a basic process in language evolution. A standard observation
is that English suffixes are largely class-changing and productive (`-ness`,
`-ment`, `-ity`, `-able`), and therefore semantically regular, whereas prefixes
are class-maintaining semantic modifiers, and many are Latinate borrowings that
arrived already lexicalized. This predicts that prefixed words should be less
compositional than suffixed words.

We propose a mechanism that subsumes the category description: **serial
position**. Retrieval and processing of a word proceed from its first element. The
first element therefore becomes the dominant cue bound to the lexical entry. When
that first element is a semantically light affix, the binding attaches to the
whole form rather than to a meaningful root, so the form is stored holistically
and is free to drift. When the first element is the root (as in a suffixed word),
the root remains the cue and keeps anchoring the word to its meaning, leaving the
suffix a predictable modifier.

This position account makes predictions beyond English grammar. It should hold in
any concatenative language (affix-first drifts more than root-first). It should
behave differently in a templatic language such as Arabic, where the anchor is a
consonantal root present throughout every form. And it should appear in an
autoregressive language model, which is forced to traverse the prefix tokens
before resolving the word, regardless of the absence of human memory.

Contributions: (1) a vector-offset compositionality measure with a bound-root
extension; (2) a validated-morphology English result and its robustness across
embeddings, time, and languages; (3) a demonstration that the asymmetry is present
in the representations of a next-token-prediction model, supporting a
substrate-independent, processing-order account.

## 2. Method

**Compositionality.** For an affixed word we compute

> comp(word) = cosine( vec(word), affix_offset(affix) + root_meaning )

where high values mean the word lands where affix and root predict (transparent)
and low values mean it does not (novel / drifted). The affix offset is the typical
semantic shift the affix imparts, learned from that affix's own transparent words
(two-pass: compute the mean shift, then recompute over the more compositional half
so drifted outliers do not poison it). This is the vector-analogy relation
(`king - man + woman = queen`) applied to morphology. We deliberately avoid
subword (fastText) vectors, which would make a word and its root artificially
similar through shared character n-grams.

We first attempted the naive comparison of `vec(word)` to `vec(root)` directly and
found it measures topic co-occurrence, not compositional meaning: it scored
`remember`/`member` and `rerun`/`run` as equally distant (0.11 each), which is
uninformative. The offset correction separates them.

**Bound roots.** A bound root such as `-ceive` (receive, perceive, conceive,
deceive) has no standalone vector. We reconstruct its meaning from its
morphological family, leave-one-out (each sibling minus its own affix offset,
excluding the word being scored). An incoherent family yields a scattered
reconstruction, so its members read as novel, which is the desired behavior.

**Data.** English morphology comes from MorphoLEX (Sánchez-Gutiérrez et al. 2018),
which provides validated prefix / root / suffix parses for about 68,000 words and
marks free versus bound roots. We use single-affix, single-root words for a clean
contrast (prefixed: one prefix, one root, no suffix; suffixed: one suffix, one
root, no prefix), collapse inflectional variants to the shortest (lemma) form, and
report both type-level and usage-frequency-weighted (token-level) rates. Word
frequencies come from the wordfreq package. We classify a word as "novel" at
compositionality below 0.15, but treat this only as a per-embedding descriptive
ruler; the comparable quantities are the prefix/suffix direction, ratio, and a
Mann-Whitney rank test on the continuous score.

Embeddings: GloVe (Wikipedia + Gigaword, 300d) and word2vec (Google News, 300d)
for English; HistWords per-decade SGNS vectors (fiction subset) for the diachronic
test; ConceptNet Numberbatch word vectors for the cross-linguistic tests; GPT-2
last-layer hidden states for the model test.

## 3. Results

### 3.1 English, validated morphology

Prefixed words are non-compositional far more often than suffixed words (GloVe):

| measure | prefix | suffix |
|---|---|---|
| type-level novelty | 24.6% | 19.0% |
| token-level (usage-weighted) | 34.7% | 21.1% |
| bound-root subset only | 37.6% | 14.2% |
| mean compositionality | +0.265 | +0.325 |

Mann-Whitney p approximately 5.7e-31 (n = 1,651 prefixed, 7,045 suffixed). The gap
widens with frequency: in the rarest band prefixed and suffixed words are nearly
equal (about 20% each), in the most common band 39.0% versus 20.7%. Rare affixed
words are about equally transparent on both sides; common prefixed words
lexicalize. This frequency-widening is the lexicalization signature and recurs in
every system below.

### 3.2 Robustness across embeddings

On word2vec (different corpus and algorithm) the absolute rates shift, because
word2vec has higher baseline cosines, but everything invariant is preserved: type
12.5% versus 7.1%, token 27.7% versus 11.8%, bound-root 9.4% versus 3.7%, p
approximately 6.2e-79, and the same frequency-widening. The conclusion travels;
the threshold is a per-corpus ruler.

### 3.3 Diachronic

Using per-decade HistWords vectors and computing compositionality inside each
decade's own space, the prefix versus suffix gap is present in every decade with
adequate coverage (1890 to 1990) and is stable at +0.04 to +0.08 (Figure 4). It is
not a modern artifact. However, an aligned semantic-change measure (orthogonal
Procrustes, 1890 to 1990) does not show prefixed words drifting faster within this
window (prefix mean change 0.478 versus suffix 0.470, p = 0.097). The asymmetry is
largely already in place by 1890. We interpret this as the generation having
occurred at or before borrowing, leaving a stable outcome rather than an ongoing
acceleration observable in a recent fiction corpus.

### 3.4 Cross-linguistic

We validated Numberbatch on the English MorphoLEX entries first; the asymmetry
survived (7.6% versus 2.6%, p approximately 1.6e-167), confirming the instrument
despite its retrofitting. For Spanish, French, Italian, and Portuguese, using
orthographic decomposition, all four show prefix far above suffix novelty with the
same frequency-widening (es 20.6/3.9, fr 15.9/6.8, it 18.1/4.2, pt 22.7/4.7; all p
< 1e-100; Figure 5). We caution that orthographic decomposition inflates the
prefix side in Romance through Latinate false splits (`invita = in + vita`), so
this is corroborative in direction only, not a clean magnitude.

### 3.5 Language-model representations

Taking each word's GPT-2 last-layer hidden state at its final sub-token (the state
after reading the whole word left to right) and applying the identical metric,
raw vectors are degenerate (mean compositionality +0.960 for both sides) due to the
known anisotropy of GPT-2 representations (Ethayarajh 2019). After the standard
correction (mean-centering and removing the top principal component; Mu and
Viswanath 2018), the asymmetry appears: prefix mean +0.313 versus suffix +0.424,
type 33.4% versus 26.4%, p approximately 1.1e-16, with the same frequency-widening
(Figure 6); it strengthens further with a second component removed (p approximately
1.4e-23). The offset vector is unchanged by centering, so the correction targets
exactly the anisotropy and not the effect.

A pure next-token predictor, with no memory and no brain, reproduces the
human-text asymmetry. This is the strongest evidence that the mechanism is
left-to-right processing order itself.

### 3.6 Family coherence (English and Arabic)

A second, independent metric agrees. Grouping English words by root and measuring
within-family mean pairwise cosine, prefix-only families are less coherent (+0.174)
than suffix-only families (+0.209), against a random baseline of +0.043: prefixed
words scatter away from their shared root more than suffixed words do. For Arabic
(templatic, no clean prefix/suffix contrast), approximate triliteral-root families
are strongly coherent (+0.447 versus a +0.025 baseline), consistent with the
consonantal root acting as a powerful anchor preserved across forms. The Arabic
figure is confounded by inflectional variants and heuristic root extraction and is
reported only as a teaser.

## 4. Discussion

The results converge on a position account of lexicalization. Within every system
we tested, affix-first words are less compositional than root-first words, the gap
grows with usage frequency, and a second coherence metric and a non-English,
non-concatenative case point the same way. The model result matters most: a system
optimized only to predict the next token, processing strictly left to right,
develops the same asymmetry in its internal representations. This decouples the
phenomenon from human memory and ties it to processing order, which is exactly what
the serial-position hypothesis predicts.

The frequency-widening is naturally read as lexicalization in action: the more a
form is retrieved, the harder its first-position binding sets, so the most-used
prefixed words are the most drifted. The diachronic data refine the claim: the
asymmetry is old and stable, so we are observing the accumulated outcome of a
process that largely completed earlier, rather than catching it accelerate in a
recent window.

## 5. Limitations

The compositionality measure is distributional and conflates meaning with topic;
the offset correction removes the systematic part, but per-word scores are noisy,
so only population comparisons are trustworthy. The Romance results use
orthographic decomposition, which inflates the prefix side via Latinate false
splits. The Arabic probe lacks a morphological analyzer and conflates inflection
with derivation. The diachronic change measure uses a single long-range alignment
that is noisy. The model test requires an anisotropy correction whose strength is a
researcher degree of freedom, although the effect is significant under any standard
setting. Finally, the design is correlational and synchronic; it does not isolate
position from category by experiment.

The cleanest confirmations would be: validated derivational databases for the
non-English languages (for example Démonette for French); a morphological analyzer
for Arabic on vocalized text; a decade-by-decade diachronic change measure on an
anchored alignment; and a design that isolates position from category, such as
comparing prefix versus suffix lexicalization within a single language's
neologisms to watch the process happen, or contrasting a heavily prefixing language
with a heavily suffixing one on matched productive morphology.

## 6. Conclusion

Prefixed words lose compositional meaning more than suffixed words, robustly in
English and in direction across embeddings, time, and languages, and the same
asymmetry emerges in the representations of an autoregressive language model. The
pattern is better explained by serial processing position than by affix category:
the first element you must traverse to reach a word becomes its anchor, and a
light first element lets the whole form drift. Lexicalization, on this view, is in
part a side effect of reading and predicting language one element at a time.

## Data and code availability

All code, the running research log (`notes.md`), figures, and per-test statistics
are at https://github.com/GenericJam/affix-novelty (MIT). Large external datasets
(MorphoLEX, HistWords, Numberbatch) are fetched by `fetch_data.sh`.

## Acknowledgments

Analysis and drafting were carried out with AI coding assistance (Claude Opus).

## References

- Bybee, J. (2010). *Language, Usage and Cognition*. Cambridge University Press.
- Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. *EMNLP*.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. *ACL*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Mu, J., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. *ICLR*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI technical report*.
- Sánchez-Gutiérrez, C. H., Mailhot, H., Deacon, S. H., & Wilson, M. A. (2018). MorphoLex: A derivational morphological database for 70,000 English words. *Behavior Research Methods*, 50(4), 1568-1580.
- Speer, R., Chin, J., & Havasi, C. (2017). ConceptNet 5.5: An open multilingual graph of general knowledge. *AAAI*.
