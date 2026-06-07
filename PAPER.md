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
candidate explanations and find the picture is multi-factor. A serial-position
account is the original motivation, but a within-English test rejects the simplest
grammatical-function variant of it (class-maintaining suffixes such as `-hood`,
`-ship` are among the most transparent, not the least), and a concreteness analysis
shows that referentiality, whether a derived word names a concrete thing, predicts
drift independently but only weakly (standardized beta about -0.12). The dominant
single predictor remains being a prefix (beta about -0.31), an effect not reducible
to concreteness or frequency and largely traceable to Latinate bound-root families.
We keep the human and language-model evidence on separate tracks, since the models
are trained on human text and so cannot independently corroborate it; a
bidirectional control (BERT) reproduces the asymmetry as strongly as an
autoregressive model (GPT-2), showing the effect is distributional and not a product
of left-to-right processing, while a nonce-word probe, which survives a control,
shows the model generalizes the pattern productively to forms it never saw. The
dependable contributions are the measured cross-linguistic outcome, the
position-versus-function dissociation, and a negative processing-order control;
the mechanism of lexicalization is left open. We discuss the implications for
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

### 3.5 Position, function, and referentiality

We tested two explanations beyond position. The first is grammatical function:
English suffixes are mostly class-changing (and so regular), prefixes
class-maintaining (and so free to drift). The Romance diminutives first looked like a
clean dissociation: class-maintaining diminutive and evaluative suffixes drift far
more than class-changing suffixes, in Spanish (28.1% versus 10.2% novel, p about
3.8e-19) and Italian (23.6% versus 8.3%, p about 8.3e-38), even exceeding the prefix
rates. But a cleaner within-English test on validated morphology rejects the simple
function hypothesis. Grouping single-affix words into prefix, class-maintaining
suffix, and class-changing suffix, the class-maintaining suffixes are the MOST
transparent, not the least (novel below 0.15 of 15.6% versus 21.0% for class-changing
versus 24.6% for prefixes). The English class-maintaining set is dominated by
regular, transparent `-ism`/`-ist`/`-hood`/`-ship`; it is not a coherent drift class.
So class-changing versus class-maintaining is not the variable.

The per-suffix gradient points to a better one that cuts across part of speech:
referentiality. Suffixes that NAME concrete things (`-er` agent or instrument,
`-age`, `-ling`, `-ster`, `-ory` place, `-ine`/`-ite`/`-ate` substances) drift;
suffixes for ABSTRACT qualities (`-ness`, `-ity`, `-ism`, `-est`) stay compositional.
We tested this directly with concreteness norms (Brysbaert et al. 2014), with no
affix hand-coding: a derived word's concreteness predicts its drift (Spearman r about
-0.13 for suffixes; suffix novelty rises from 14.2% to 23.1% across the abstract-to-
concrete tertiles). In a standardized regression controlling for frequency and side,
concreteness has a real but small independent effect (beta about -0.12), being a
prefix is the dominant predictor (beta about -0.31), and frequency is weakly toward
transparency (beta about +0.05). So the Romance diminutive effect is real but is
better read as referentiality (a diminutive names a specific thing, e.g. Spanish
`bolsillo` "pocket") than as class-maintenance. The synchronic outcome is
multi-factor: the prefix penalty dominates and is not explained by concreteness or
frequency (it traces to the Latinate bound-root families of Section 3.1), with
referentiality a smaller independent contributor. Caveat: the orthographic Romance
and short-diminutive decompositions inflate magnitudes via false splits; the
within-English and concreteness tests do not depend on them.

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

A separate probe asks whether GPT-2 GENERALIZES the asymmetry to novel words it never
saw, which would indicate a productive learned bias rather than inherited per-word
meanings. On 785 novel affix-plus-root combinations, all genuinely compositional by
construction (`unsmall` = not small; `jumpness` = state of jumping), GPT-2 represents
the novel prefixed forms as much less compositional than the novel suffixed forms
(mean +0.40 versus +0.69, p about 1e-37). The obvious confound is that the offsets
are learned from real words whose prefix side is more opaque, which could mechanically
depress novel-prefix scores. A control rejects it: relearning each affix's offset from
only its most-transparent real words, and sweeping how aggressively we filter (top
50%, 25%, 10% by compositionality), the gap does not shrink toward zero but holds and
slightly grows (gap +0.29 to +0.36, p < 1e-34 throughout). So GPT-2 represents novel
prefixed forms as non-compositional even relative to the cleanest prefix shift: it has
generalized "prefixed forms tend to be opaque" to words it never saw. This is a
productive inductive bias, not per-word memorization, though its origin is still the
human distribution the model was trained on. A bidirectional version of the probe
settles the remaining checks: BERT (WordPiece) shows the novel-form gap even more
strongly than GPT-2 (gap +0.39 versus +0.29, p < 1e-100) and it survives the same
control, so the bias is architecture-independent (a model that never reads left to
right has it too) and not a single-tokenizer artifact.

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

On mechanism, the simple single-variable stories all fail. The serial-position
account motivated the study, but the tests undercut it as the synchronic
explanation. Its grammatical-function variant is rejected within English: the
class-maintaining suffixes are the most transparent, not the least, so it is not
class-change that protects compositionality. What does carry a real, if small,
independent signal is referentiality: concrete-naming derivations drift more than
abstract ones (concreteness beta about -0.12), which reinterprets the Romance
diminutive effect (a diminutive names a concrete thing) without supporting the
function account. The dominant factor is simply being a prefix (beta about -0.31),
unexplained by concreteness or frequency and traceable to the Latinate bound-root
families. The synchronic outcome is multi-factor, with no single variable
sufficient.

We keep the human and language-model evidence on separate tracks. The models are
trained on human text, so an LLM reproducing the asymmetry is inheritance, not
independent corroboration; lumping the two would double-count. On its own the LLM
track says the asymmetry is distributional, since a bidirectional model shows it as
strongly as an autoregressive one and so it is not a processing-order signature; and
a nonce-word probe, which survives a control that rejects the obvious offset
confound, shows the model generalizes the pattern productively to words it never saw.
That probe is the one place the LLM track reports something beyond per-word
inheritance: the prefix-opacity regularity is encoded as a generalizing inductive
bias, not just memorized lexeme by lexeme.

This clarifies what distributional embeddings can and cannot show. Every embedding,
static or contextual, autoregressive or bidirectional, measures the present
lexicalized state of the language, not the historical process that produced it. The
serial-position hypothesis is a claim about that process, and our synchronic
measurements cannot adjudicate it. The diachronic data are consistent with the
outcome being old and stable (already in place by 1890) but show no recent-century
acceleration, so they do not catch the process in motion either.

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
models. The explanation is multi-factor rather than any single variable. The
grammatical-function hypothesis (class-changing versus class-maintaining) is rejected
within English; referentiality, whether a derived word names a concrete thing, has a
real but small independent effect that reinterprets the Romance diminutives; and the
dominant factor is simply being a prefix, an effect traceable to Latinate bound-root
families and not reducible to concreteness or frequency. We keep the human and
language-model evidence separate, because the models are trained on human text and so
inherit rather than corroborate the pattern; a bidirectional control shows the
model asymmetry is distributional, not a product of left-to-right processing. The
serial-position hypothesis that motivated the work is not supported as the synchronic
mechanism, though it remains a live hypothesis about the historical process that our
distributional measurements cannot reach. The dependable contributions are the
measured cross-linguistic outcome, the position-versus-function dissociation, and a
negative processing-order control; the mechanism of lexicalization is left open for
diachronic and behavioral work.

## Data and code availability

All code, the running research log (`notes.md`), figures, and per-test statistics
are at https://github.com/GenericJam/affix-novelty (MIT). Large external datasets
(MorphoLEX, HistWords, Numberbatch) are fetched by `fetch_data.sh`.

## Acknowledgments

Analysis and drafting were carried out with AI coding assistance (Claude Opus).

## References

- Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods*, 46(3), 904-911.
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
