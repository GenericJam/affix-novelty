# When do affixed words lose their parts? A prefix-suffix asymmetry, a diminutive counterexample, and a negative autoregressive control

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
century and replicates in direction across four Romance languages. We then test
two competing accounts. The serial-position account is challenged by a
counterexample we report: class-maintaining diminutive suffixes drift far more
than class-changing suffixes in Spanish and Italian, dissociating affix position
from grammatical function and favoring function. A processing-order reading is
tested with language models and fails a control: the asymmetry appears equally in
an autoregressive model (GPT-2) and a bidirectional model (BERT), so it reflects
distributional structure already present in the corpus rather than left-to-right
processing. The robust result is the outcome (prefixed and class-maintaining
affixes are less compositional); grammatical function predicts it better than
serial position; and distributional embeddings measure the lexicalized state of a
language, not the mechanism that produced it. We discuss the implications for
studying lexicalization in language evolution.

## 1. Introduction

Lexicalization, the hardening of a composed form into a single stored unit with
its own meaning, is a basic process in language evolution. A standard observation
is that English suffixes are largely class-changing and productive (`-ness`,
`-ment`, `-ity`, `-able`), and therefore semantically regular, whereas prefixes
are class-maintaining semantic modifiers, and many are Latinate borrowings that
arrived already lexicalized. This predicts that prefixed words should be less
compositional than suffixed words.

We consider two accounts of this asymmetry and try to separate them. The first is
a **serial-position** account: retrieval and processing of a word proceed from its
first element, which becomes the dominant cue bound to the lexical entry; when that
first element is a semantically light affix the form is stored holistically and is
free to drift, whereas when the first element is the root (a suffixed word) the
root keeps anchoring the meaning. This has a clear ancestor in the processing
explanation of the cross-linguistic suffixing preference (Cutler, Hawkins &
Gilligan 1985), which argues prefixes are costly precisely because they precede the
stem; we extend that processing/typology argument to a prediction about the
semantic drift of existing words. The second account is **grammatical function**:
English suffixes happen to be class-changing (and therefore regular), while
prefixes are class-maintaining meaning-modifiers free to drift, so the real driver
is whether the affix performs a regular class-changing job, independent of
position. In English the two accounts are confounded, because the suffixes are the
class-changing ones.

The accounts come apart in two places we test. (i) **Class-maintaining suffixes**
(diminutives), which are suffixes that do not change category. Serial position
predicts they stay transparent (root first); grammatical function predicts they
drift like prefixes. (ii) **Language-model architecture**. If left-to-right
processing is what matters, an autoregressive model should show the asymmetry more
than a bidirectional one; if the asymmetry is merely distributional, both should
show it equally.

We note up front that the vector-offset compositionality metric is not new
(Lazaridou et al. 2013), and that a prefix-versus-suffix transparency difference is
foreshadowed in distributional semantics (Lazaridou et al. 2013; Stupak & Baayen
2022 for German particle verbs). Our contributions are: (1) a systematic
validated-morphology measurement of the asymmetry in English, with a bound-root
extension, robust across embeddings, time, and four Romance languages; (2) a
counterexample dissociating position from function via Romance diminutive
suffixes; (3) a negative language-model control showing the asymmetry is
distributional, not a signature of autoregressive processing.

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

### 3.5 Position versus function: a diminutive counterexample

The serial-position and grammatical-function accounts are confounded in English. We
separate them with class-maintaining suffixes: suffixes (root-first, so serial
position predicts transparency) that do not change category (so grammatical function
predicts drift). Holding position constant and varying function, diminutive and
evaluative suffixes drift far more than class-changing suffixes, in both Spanish
(28.1% versus 10.2% novel, p approximately 3.8e-19) and Italian (23.6% versus 8.3%,
p approximately 8.3e-38). The diminutive suffixes even exceed the prefix novelty
rates in the same languages, so a suffix subtype out-generates prefixes. This favors
grammatical function over pure position. Caveat: short diminutive affixes attract
more orthographic false splits than long class-changing affixes, so the magnitude is
inflated, but the direction is large, significant in two languages, and matches the
textbook status of diminutives as lexicalization machines (Jurafsky 1996).

### 3.6 Language models, and a bidirectional control

Taking each word's GPT-2 hidden state and applying the identical metric, raw vectors
are degenerate (mean compositionality +0.960 for both sides) due to the known
anisotropy of GPT-2 representations (Ethayarajh 2019). After the standard correction
(mean-centering and removing the top principal component; Mu and Viswanath 2018) the
asymmetry appears (prefix less compositional than suffix, p approximately 1e-16 to
1e-23 depending on components removed). The offset vector is unchanged by centering,
so the correction targets the anisotropy, not the effect.

The tempting interpretation is that an autoregressive model, forced to read through
the prefix before resolving the word, reproduces the effect because of left-to-right
processing. A control refutes this. Comparing GPT-2 (autoregressive) with BERT
(bidirectional) under identical extraction (mean-pooled sub-token hidden states,
same anisotropy correction), the asymmetry is essentially as strong in BERT as in
GPT-2: prefix/suffix novelty ratio 1.57 versus 1.62, gap +0.118 versus +0.134, both
p < 1e-50. A bidirectional model that never traverses the word left to right shows
the asymmetry just as strongly. We conclude that the asymmetry is a distributional
property of the corpus, already present in static embeddings (Lazaridou et al.
2013), and not a signature of autoregressive processing. The language-model
results, like the static embeddings, measure the lexicalized outcome, not the
mechanism that produced it.

### 3.7 Family coherence (English and Arabic)

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

The robust result is the outcome, not a particular mechanism. Across every system
we tested, affix-first (prefixed) words are less compositional than root-first
(suffixed) words, the gap grows with usage frequency, and an independent
family-coherence metric agrees. This outcome is solid and, in direction, was
already foreshadowed by distributional semantics (Lazaridou et al. 2013) and by the
processing account of the suffixing preference (Cutler, Hawkins & Gilligan 1985).

On mechanism, our two diagnostic tests pull against the simplest serial-position
story. First, the diminutive counterexample: class-maintaining suffixes drift far
more than class-changing suffixes, even out-drifting prefixes, with position held
constant. This says grammatical function (whether the affix does a regular
class-changing job) is doing real work that pure position cannot explain. Second,
the bidirectional control: BERT reproduces the asymmetry as strongly as GPT-2, so
the effect is distributional rather than a product of left-to-right processing.
Together these favor a function-based reading of the synchronic outcome and remove
the language model as evidence for a processing-order mechanism.

This clarifies what distributional embeddings can and cannot show here. Every
embedding, static or contextual, autoregressive or bidirectional, measures the
present lexicalized state of the language; none of them observes the historical
process that produced it. The serial-position hypothesis is a claim about that
process (how forms lexicalize over time), and our synchronic measurements cannot
adjudicate it. The diachronic data are consistent with the outcome being old and
stable (already in place by 1890), but the recent-century window shows no
acceleration, so it does not catch the process in motion either.

## 5. Limitations

The compositionality measure is distributional and conflates meaning with topic;
the offset correction removes the systematic part, but per-word scores are noisy,
so only population comparisons are trustworthy. The Romance results use
orthographic decomposition, which inflates the prefix side via Latinate false
splits (the diminutive magnitudes are inflated the same way, though the direction is
robust). The Arabic probe lacks a morphological analyzer and conflates inflection
with derivation; the agglutinative (Turkish/Finnish) test was likewise too noisy
without analyzers to be conclusive. The diachronic change measure uses a single
long-range alignment that is noisy. The model test requires an anisotropy
correction whose strength is a researcher degree of freedom, although the effect,
and the BERT-versus-GPT-2 equivalence, hold under any standard setting. Most
fundamentally, every measurement here is synchronic and distributional: it captures
the lexicalized state, not the historical process, so it cannot directly test the
serial-position mechanism.

The work that would move this forward: validated derivational databases for the
non-English languages (for example Démonette for French) and morphological
analyzers for Arabic, Turkish, and Finnish, to clean up the cross-linguistic
magnitudes; a decade-by-decade diachronic change measure on an anchored alignment;
and, to get at the mechanism rather than the outcome, behavioral or diachronic
designs, such as tracking prefix versus suffix lexicalization in neologisms from
their coining date, where the process can actually be observed.

## 6. Conclusion

Prefixed words lose compositional meaning more than suffixed words, robustly in
English and in direction across embeddings, time, and four Romance languages, and
the same asymmetry is present in both autoregressive and bidirectional language
models. Two diagnostic tests then constrain the explanation. A diminutive
counterexample shows that class-maintaining suffixes drift more than class-changing
ones, so grammatical function, not affix position alone, drives the synchronic
pattern. A bidirectional control shows the language-model asymmetry is distributional
rather than a product of left-to-right processing. The serial-position hypothesis,
the original motivation, is not supported by these tests as the synchronic
mechanism, though it remains a live hypothesis about the historical process that
our distributional measurements cannot reach. The dependable contribution is the
measured outcome and the position-versus-function dissociation; the mechanism of
lexicalization, and whether processing order contributes to it over historical
time, is left open for diachronic and behavioral work.

## Data and code availability

All code, the running research log (`notes.md`), figures, and per-test statistics
are at https://github.com/GenericJam/affix-novelty (MIT). Large external datasets
(MorphoLEX, HistWords, Numberbatch) are fetched by `fetch_data.sh`.

## Acknowledgments

Analysis and drafting were carried out with AI coding assistance (Claude Opus).

## References

- Bybee, J. (2010). *Language, Usage and Cognition*. Cambridge University Press.
- Cutler, A., Hawkins, J. A., & Gilligan, G. (1985). The suffixing preference: a processing explanation. *Linguistics*, 23(5), 723-758.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. *EMNLP*.
- Hay, J. (2001). Lexical frequency in morphology: is everything relative? *Linguistics*, 39(6), 1041-1070.
- Jurafsky, D. (1996). Universal tendencies in the semantics of the diminutive. *Language*, 72(3), 533-578.
- Lazaridou, A., Marelli, M., Zamparelli, R., & Baroni, M. (2013). Compositionally derived representations of morphologically complex words in distributional semantics. *ACL*.
- Marelli, M., & Baroni, M. (2015). Affixation in semantic space: modeling morpheme meanings with compositional distributional semantics. *Psychological Review*, 122(3), 485-515.
- Stupak, I., & Baayen, R. H. (2022). An inquiry into the semantic transparency and productivity of German particle verbs and derivational affixation. *The Mental Lexicon*, 17(3), 422-457.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. *ACL*.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv:1301.3781*.
- Mu, J., & Viswanath, P. (2018). All-but-the-top: Simple and effective postprocessing for word representations. *ICLR*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI technical report*.
- Sánchez-Gutiérrez, C. H., Mailhot, H., Deacon, S. H., & Wilson, M. A. (2018). MorphoLex: A derivational morphological database for 70,000 English words. *Behavior Research Methods*, 50(4), 1568-1580.
- Speer, R., Chin, J., & Havasi, C. (2017). ConceptNet 5.5: An open multilingual graph of general knowledge. *AAAI*.
