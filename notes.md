# Research log: serial position and the lexicalization of affixed words

Running notes for the affix-novelty project. Newest sections appended at the
bottom. This file documents the process, the dead ends, and the reasoning, not
just the polished result.

## The question, and the field it belongs to

The field of interest is **language evolution**: how words are generated and how
their meanings drift over time, treated as an evolutionary / dynamical process,
not as static linguistic description.

Starting question: are prefixes more likely than suffixes to generate "new
words", meaning words whose meaning has become independent of their parts even
though their origin is transparent? `remember` started as roughly re + member
("bring the members back together") and is now only about memory. `rerun` never
did that; it still just means "run again."

## The deeper hypothesis (Kevin)

The prefix-vs-suffix framing is a proxy for a hypothesis about **serial
position and memory retrieval**:

> The prefix is the first thing you have to remember in order to remember the
> word. Because it is first, it becomes the retrieval cue, the handle that
> attaches the word to a knowledge node in the brain. That permanent
> first-position association is what lets the whole form lexicalize into a single
> stored unit with its own meaning.

Mechanism, stated as a chain:
1. Retrieval is serial / cue-first. You traverse the first morpheme to reach the word.
2. The first morpheme therefore becomes the strongest cue bound to the lexical entry.
3. When the first morpheme is a semantically light affix (re-, con-, de-), the
   binding is to the *whole form*, not to a meaningful root, so the form is
   stored holistically and is free to drift.
4. When the first morpheme is the *root* (as in a suffixed word), the root stays
   the cue and keeps anchoring the word to the root's meaning, so the word stays
   compositional and the suffix remains a predictable modifier.

Universal prediction across concatenative languages: **affix-first (prefixed)
words drift more than root-first (suffixed) words**, because the variable that
matters is which element is first, not which kind of affix it is.

### Extension to LLMs

Autoregressive language models predict the next token strictly left to right.
They are *forced* to traverse the prefix tokens before resolving the word. If the
serial-position mechanism is real and not specific to human memory, then an LLM
trained on next-token prediction should also represent prefixed words more
holistically (less compositionally) than suffixed words. Testing this turns a
claim about brains into a claim about a mechanism that any left-to-right
sequence predictor would exhibit, which is exactly the kind of substrate-
independent result that matters for language evolution.

## Method development (English, the anchor study)

This took several iterations. The dead ends are informative.

### What "drift" means, operationally

We measure non-compositionality distributionally. For an affixed word:

```
compositionality(word) = cosine( vec(word),  affix_offset(affix) + root_meaning )
```

high = the word lands where affix + root predict (transparent);
low  = it does not (novel / drifted).

### Iteration 1: raw word-vs-root cosine. FAILED.

First attempt compared `vec(word)` to `vec(root)` directly. This measures topic
co-occurrence, not compositional meaning. Concrete failure from the sanity check:

```
remember vs member  cos = 0.11   (drifted, we want LOW)   <- ok
rerun    vs run     cos = 0.11   (transparent, want HIGH) <- identical, useless
```

A TV `rerun` occurs in different contexts than jogging `run`, so it scores far
from `run` despite being perfectly compositional. Raw cosine cannot tell topic
drift from semantic drift. Abandoned.

### Iteration 2: affix-offset (vector analogy). WORKS.

Learn the typical shift an affix imparts, the same idea as
`king - man + woman = queen`. For prefix re-, learn `offset = mean(vec(derived) -
vec(root))` over transparent re- words (redo, reopen, rebuild...). Then compare
the word to `root + offset`. This separated the cases:

```
                 raw cos   compositionality(root+offset)
remember           0.11        -0.09   (novel, correct)
report             0.15        -0.03   (novel, correct: re+port "carry back")
rerun              0.11         0.23   (more transparent, correct direction)
reopen             0.41         0.52   (transparent, correct)
rebuild            0.59         0.65   (transparent, correct)
```

The offset is learned two-pass (compute mean, then recompute using only the more
compositional half) so drifted outliers do not poison the learned shift.

Also decided: **do NOT use fastText / subword vectors.** They build a word vector
partly from its character n-grams, so `remember` and `member` would look similar
just from shared letters, destroying the signal. Use word-level vectors (GloVe,
word2vec).

### Iteration 3: orthographic decomposition. CONTAMINATED.

Decomposed words by string matching (strip affix, require the remainder to be a
real standalone word). The offset metric gave prefix ~25% vs suffix ~13% novel,
but inspection showed heavy contamination by orthographic coincidences:
`color = co + lor`, `surgery = ery + surge`, `never = nev + er`. These false
splits score as "novel" because there is no real root relationship, inflating
both sides unevenly. Directionally suggestive, not trustworthy.

### Iteration 4: validated morphology (MorphoLEX). TRUSTWORTHY.

Switched to **MorphoLEX** (Sanchez-Gutierrez et al. 2018), which ships validated
morpheme parses for ~68k English words, marking prefixes / roots / suffixes and
distinguishing free vs bound roots. Two wins:
1. Removes the false-split problem (it knows `color` is monomorphemic).
2. Brings **bound-root** prefix families into scope: `-ceive` (receive, perceive,
   conceive, deceive), `-duct` (reduce, produce, conduct), `-tain`, `-sume`.
   These are exactly the most-drifted prefix words and orthography could never
   reach them (the root is not a standalone word).

Contrast set: pure single-affix, single-root words. Prefix = 1 prefix, 1 root,
0 suffix. Suffix = 0 prefix, 1 root, 1 suffix.

### Bound-root metric

A bound root (`-ceive`) has no standalone vector, so `root_meaning` is
reconstructed from the root's morphological family, leave-one-out (each sibling
minus its own affix offset, excluding the word itself so it never predicts
itself). An incoherent family like `-ceive` (receive/conceive/deceive all mean
unrelated things) reconstructs to a scattered centroid, so each member reads as
novel. This also fixes a collision: `reduce`'s etymological root `duct` happens
to coincide with the free English word "duct" (a tube), which is semantically
irrelevant; reconstructing from the family beats trusting the tube vector.

### Cleanups

- Dedup inflectional variants to one lexeme, keeping the **shortest** surface
  form (the lemma), which also gives the cleanest vector (less tense/topic).
- **Token weighting**: report novelty weighted by usage frequency (wordfreq), not
  just type counts, since the hypothesis is about words people actually retrieve.

## Results so far (English, GloVe Wikipedia+Gigaword 300d)

```
measure                              prefix   suffix
type-level novelty (comp<0.15)        24.8%    19.0%
token-level (usage-weighted)          35.0%    21.1%
bound-root subset only                38.8%    14.2%
mean compositionality                +0.264   +0.325
Mann-Whitney U on compositionality        p = 2e-31
```

Prefixes generate non-compositional words ~1.3x (type) to 2.7x (bound-root) as
often as suffixes. Holds in every frequency band, and the gap **widens with
frequency**: in the rarest band the two are nearly equal (20.5% vs 19.5%), in the
commonest band 39.7% vs 20.7%. Rare affixed words are about equally transparent
on both sides; what differs is that *common* prefixed words lexicalize. This is
the lexicalization signature, and it is consistent with the serial-position story
(the words you retrieve most are the ones whose first-position binding hardens).

Repo: https://github.com/GenericJam/affix-novelty (MIT, public).

## Follow-up tests planned (this phase)

Reframed around the serial-position hypothesis, not just prefix-vs-suffix.

1. **Second embedding / corpus** (robustness). Re-run the English metric on a
   different-corpus, different-algorithm embedding (word2vec Google News) to show
   the asymmetry is not a GloVe artifact.
2. **Diachronic** (the language-evolution core). Use per-decade historical
   embeddings (HistWords) to test whether prefixed words actually drifted *more
   over time* than suffixed words, i.e. observe the generation process, not just
   the present-day snapshot.
3. **Romance** (cross-linguistic replication + position test). Spanish, French,
   Italian, Portuguese share the Latin prefix families (recevoir / percevoir /
   concevoir). The position hypothesis predicts the same prefix>suffix asymmetry.
4. **Arabic** (the hard, interesting case). Templatic / root-and-pattern
   morphology where the "first thing" is a discontinuous consonantal root. Tests
   whether the prediction even has a clean form when morphology is not
   concatenative. Expected to be preliminary.
5. **LLM** (substrate-independence). Run the same compositionality metric on an
   autoregressive model's word representations. Does a next-token predictor show
   the same prefix>suffix non-compositionality?

Predictions, so they are on record before running:
- (1) asymmetry replicates, similar magnitude.
- (2) prefix words show larger semantic change per unit time; effect strongest in
  mid/high frequency.
- (3) replicates in all four Romance languages; possibly stronger (more shared
  Latinate lexicalization).
- (4) unclear; if the root stays the anchor regardless of position, Arabic
  derivation should be relatively transparent, which would support "anchor =
  whatever the root-cue is" over "anchor = literally first segment."
- (5) replicates in the LLM, supporting a substrate-independent serial-position
  mechanism.

## Test 1 result: second embedding (word2vec Google News). REPLICATES.

Ran the identical metric on word2vec-google-news-300 (different corpus, different
algorithm) versus GloVe Wikipedia+Gigaword.

```
                       GloVe (type / token / bound)   word2vec (type / token / bound)
prefix novel<0.15        24.6% / 34.7% / 37.6%           12.5% / 27.7% /  9.4%
suffix novel<0.15        19.0% / 21.1% / 14.2%            7.1% / 11.8% /  3.7%
mean comp pref/suff       +0.265 / +0.325                 +0.348 / +0.440
Mann-Whitney p              5.7e-31                          6.2e-79
```

Key lesson: the **absolute** novelty rate at a fixed threshold is NOT comparable
across embeddings, because word2vec has higher baseline cosines (everything looks
more compositional), so the 0.15 cut catches fewer words. What IS invariant, and
what we should report as the result, is (a) the direction, (b) the prefix/suffix
ratio (1.8x to 2.5x in both), (c) the frequency-widening pattern (replicates:
word2vec prefix 9.6% to 23.3% across bands vs suffix 5.2% to 10.2%), and (d) the
rank test, which is actually stronger on word2vec. Conclusion travels; threshold
is a per-corpus ruler. Going forward, lead with ratio + rank test, treat the
threshold as descriptive only.

Refactor note: extracted the scoring into `core.py` (shared by all tests),
`morpholex_entries.py` (English loader), `run_embeddings.py` (driver). GloVe
numbers reproduce the monolithic script (24.6 vs 24.8, n differs by 7 from a
minor filter-order change, negligible).

## Test 2 result: diachronic (HistWords fiction, by decade). NUANCED.

Used HistWords per-decade SGNS embeddings (fiction subset, 1800-1990). The
compositionality metric is computed inside each decade's own space, so no
cross-decade alignment is needed for the trajectory. Coverage of our modern
MorphoLEX word list in 19th-century fiction is thin, so only 1890-1990 has enough
prefix words (>=50) to be reliable.

Two findings, one confirming and one complicating:

**(a) The gap is ancient and stable.** In every observable decade prefix words
are less compositional than suffix words, by a steady +0.04 to +0.08:

```
decade  prefix  suffix  gap
1890    +0.237  +0.313  +0.076
1920    +0.193  +0.262  +0.068
1950    +0.206  +0.273  +0.067
1990    +0.192  +0.251  +0.058
```

The asymmetry is not a modern artifact. It was already fully present in 1890 and
holds flat for a century (see `fig4_diachronic.png`).

**(b) But we do NOT catch prefixes drifting faster within the window.** Aligned
semantic-change magnitude 1890 to 1990 (orthogonal Procrustes, then 1-cos):
prefix mean change 0.478 vs suffix 0.470, Mann-Whitney p = 0.097 (one-sided, not
significant). If anything, over the 20th century the suffix line dropped slightly
more (suffix -0.062, prefix -0.045), i.e. suffix words lexicalized a touch faster
in this recent window.

**Interpretation (important for the write-up).** The prefix lexicalization that
produces the asymmetry mostly happened *before* 1890. Most of these prefixed
words are old Latinate borrowings that arrived already drifted (or drifted in
Latin/Old French), so within a 1890-1990 fiction window there is little drift
left to observe. This refines the original claim: it is not that prefixes are
*currently* drifting faster, it is that the prefix-first structure produced
lexicalized forms historically and the result is stable. To observe the
generation *process* directly you would need older data, or to track neologisms
from their moment of coining, or a corpus like COHA reaching further back. The
snapshot asymmetry is real and old; the change-rate over the recent century is
roughly symmetric. Prediction (2) is therefore only partially supported: the
persistence is confirmed, the "faster drift now" part is not.

Method caveat: a single 100-year Procrustes rotation is noisy (all words show
large apparent change, ~0.47), which may swamp a modest prefix/suffix difference.
A decade-by-decade incremental alignment on a stable high-frequency anchor set
would be a cleaner follow-up if we want to push the change-rate question.

## Test 3 result: cross-linguistic (Numberbatch). DIRECTION HOLDS, MAGNITUDE CONFOUNDED.

Used ConceptNet Numberbatch word vectors (word-level, multilingual, one 3.2GB
download stream-filtered to en/es/fr/it/pt/ar). Chose it over fastText (subword
n-grams would inflate root-word similarity) and over per-language hunting.

**Instrument validation (important).** Numberbatch is retrofitted on ConceptNet,
which includes some derivation edges, so it could in principle pull derived words
toward their roots and wash out our signal (a bias *against* the hypothesis).
First ran the validated English MorphoLEX entries through Numberbatch-en: prefix
7.6% vs suffix 2.6% (type), 16.0% vs 1.8% (bound-root), p = 1.6e-167. The
asymmetry survives, so the instrument is trustworthy and any retrofit bias is
conservative. (Absolute rates are low again because Numberbatch has high baseline
cosines; direction + ratio is the comparable quantity.)

**Romance (orthographic decomposition).** All four languages show prefix >>
suffix novelty, with the same frequency-widening pattern, all p < 1e-100:

```
lang   prefix  suffix   (type novel<0.15)
es     20.6%   3.9%
fr     15.9%   6.8%
it     18.1%   4.2%
pt     22.7%   4.7%
```

**The honest problem.** Inspecting the most-novel prefix words exposes heavy
false-split contamination, and it is *prefix-specific* in Romance: `inventa =
in+venta`, `intimo = in+timo`, `profane = pro+fane`, `invita = in+vita`,
`repente = re+pente`. These are Latinate words (invitare, profanus, intimus) that
merely begin with a prefix-shaped string whose remainder coincidentally is a real
word. Romance vocabulary is saturated with in-/re-/pro-/con-/ex- Latinate forms,
so orthographic decomposition manufactures far more false prefix splits than
false suffix splits (suffix strings like -cion/-mento are longer and more
distinctive). This inflates the prefix side specifically.

Conclusion for Romance: the direction is consistent across all four languages AND
consistent with the validated English instrument, so it is corroborative. But the
magnitude is untrustworthy and I cannot fully exclude that the artifact
contributes to the direction. The proper fix is a validated derivational database
per language (Demonette for French, etc.), which is a project in itself. Logged
as suggestive, not clean. See `fig5_crosslang.png` (caveat is in the caption).

Bound-root novelty is `nan` for Romance because orthographic decomposition only
finds free roots, so it never reaches the bound-root families (Spanish
recibir/percibir/concebir, the -ceive cousins) that are the most interesting
cases. Another reason the Romance test is weaker than English.

## Test 4 result: LLM representations (GPT-2). CONFIRMS substrate-independence.

The serial-position hypothesis predicts that an autoregressive (next-token)
model, forced to read left to right through the prefix before resolving the word,
should represent prefixed words more holistically than suffixed words. Test: take
each word's GPT-2 last-layer hidden state at its final sub-token (the state after
reading the whole word, on the same MorphoLEX entries), then run the identical
compositionality metric.

**Anisotropy had to be handled first.** Raw GPT-2 hidden states are degenerate
for cosine: mean compositionality +0.960 for both sides, no signal. This is the
well-known representation-degeneration / anisotropy of GPT-2 (a single dominant
direction dominates all cosines), not evidence against the hypothesis. Applying
the standard fix (mean-center, then project out top principal components) the
asymmetry emerges and sharpens monotonically:

```
condition                         prefix  suffix   p           type novel<0.15
raw (anisotropic)                 +0.960  +0.959   4e-8        1.0% / 1.0%
centered                          +0.262  +0.310   0.054       37.6% / 35.1%
centered + top-1 PC removed       +0.313  +0.424   1.1e-16     33.4% / 26.4%
centered + top-2 PCs removed      +0.357  +0.435   1.4e-23     25.3% / 17.5%
```

Note the offset vector is unchanged by centering (it is a difference of
vectors); only the final cosine changes, which is exactly the anisotropy we
needed to remove. The frequency-widening pattern replicates here too (prefix
rises across bands faster than suffix). Reported primary = top-1 PC removed (the
minimal, standard correction); robust at top-2.

Caveat: removing PCs is a researcher degree of freedom, and the effect grows as
more are removed, so this is not a knob to lean on hard. The honest reading: raw
GPT-2 is uninformative due to a known artifact; under any standard
de-anisotropization the prefix<suffix asymmetry is present and significant, and
it matches the human-embedding result in direction, magnitude region, and the
frequency-widening shape (see `fig6_llm.png`).

Interpretation: a system trained only to predict the next token, with no memory
and no brain, develops the same prefix-vs-suffix non-compositionality. That is
strong support for the mechanism being about serial left-to-right processing
order itself, not about anything specific to human cognition. This is the most
novel result of the project for a language-evolution framing.

Also note a metric edge case: when every string can be embedded (an LLM embeds
any token sequence), `root_free` is always true, so the bound-root subset is
empty (`nan`) for the LLM and Romance runs. The family reconstruction still runs;
only the bound-root *reporting* line is N/A there.

## Test 5 result: Arabic (templatic) + English family coherence. PRELIMINARY.

Arabic is nonconcatenative (root-and-pattern), so there is no clean prefix/suffix
contrast for the main metric. Instead we tested the prediction the hypothesis
makes for a language whose anchor is a consonantal root present in every form:
derivational families should stay semantically coherent. Metric = within-family
mean pairwise cosine ("coherence") vs a random-pair baseline.

English (GloVe), grouping MorphoLEX entries by root into pure prefix vs pure
suffix families:
```
random-pair baseline           +0.043
prefix-only families (n= 42)   +0.174   (lift +0.131)
suffix-only families (n=538)   +0.209   (lift +0.166)
```
Bonus confirmation via a second metric: prefix families are LESS coherent than
suffix families, i.e. prefixed words scatter away from their shared root more
than suffixed words do, exactly as the anchor idea predicts. Modest gap, same
direction as everything else.

Arabic (Numberbatch ar), grouping ~16k frequent words by an approximate
triliteral consonant skeleton:
```
random-pair baseline           +0.025
within-root families (n=860)   +0.447   (lift +0.422)
```
Arabic root families are very strongly coherent, consistent with the consonantal
root being a powerful anchor that keeps the family semantically bound. Example
families looked real (q-b-l: before/tribe/meet/accept; m-th-l: like/example/
represent).

**Honest caveats (this is a teaser, not a clean result):**
- The skeleton heuristic lumps inflectional variants of ONE lexeme (يبعد/أبعد) in
  with genuine derivations, which inflates Arabic coherence. English families are
  deduped distinct lexemes, so the Arabic-vs-English magnitudes are NOT
  comparable.
- Different embeddings (Numberbatch ar vs GloVe en) and different baselines.
- Heuristic root extraction (no analyzer, unvocalized text) is noisy; weak roots,
  hamza, assimilation all mishandled.
A rigorous Arabic study needs a morphological analyzer (CAMeL Tools / Farasa) to
separate derivation from inflection and extract true roots, ideally on vocalized
text. What we have is directionally consistent with "the root is the anchor", and
worth a proper follow-up, nothing more.

## Synthesis: the shape of the argument

The reframing from "prefix vs suffix" to "serial position / first-element-as-
retrieval-anchor" is what makes the separate tests cohere. The claim is not about
a grammatical category, it is about processing order: the first element you must
traverse to reach a word becomes the binding cue, and when that first element is a
semantically light affix the whole form lexicalizes and drifts.

How each test bears on it, and how much weight it can hold:

| test | result | weight |
|---|---|---|
| English, validated morphology (MorphoLEX) | prefix > suffix non-compositionality, type/token/bound-root, all freq bands, p~1e-31 | STRONG (the anchor study) |
| Second embedding (word2vec) | same direction, larger ratio, p~1e-79 | STRONG (not a GloVe artifact) |
| LLM (GPT-2 hidden states) | same asymmetry after standard anisotropy fix, p~1e-16 | STRONG and novel (substrate-independent) |
| English family coherence | prefix families scatter from the root more than suffix families | SUPPORTING (second metric, same direction) |
| Diachronic (HistWords) | the gap is old and stable; generation predates the corpus window | SUPPORTING but COMPLICATING (no faster drift caught now) |
| Romance (es/fr/it/pt) | same direction x4, but orthographic decomposition inflates the prefix side via Latinate false splits | SUGGESTIVE only |
| Arabic | root families strongly coherent (root as anchor); confounded by inflation | TEASER only |

The load-bearing evidence is English-validated plus the LLM. The single most
interesting result for language evolution: a next-token predictor with no memory
and no brain reproduces the asymmetry, which says the mechanism is about
left-to-right processing order itself, exactly Kevin's "you have to drive through
the prefix to get to the word." The frequency-widening pattern (the gap grows for
common words, in every system tested) is the lexicalization signature: the more a
form is retrieved, the harder its first-position binding sets.

The honest tensions, which the write-up must keep: (1) the diachronic data shows
the asymmetry is largely *already baked in* by 1890, so we observe a stable
outcome rather than catching prefixes drifting faster in the modern window; the
"generation" mostly happened at or before borrowing. (2) Romance and Arabic are
corroborative in direction but methodologically confounded, so they are support,
not proof. (3) The fixed novelty threshold is a per-embedding ruler; the
comparable quantities are direction, ratio, and the rank test.

What a rigorous publication would still need: validated derivational databases for
the non-English languages (Demonette etc.), a morphological analyzer for Arabic,
a cleaner decade-by-decade diachronic change measure on an anchored alignment,
and ideally a controlled test that isolates POSITION from CATEGORY (e.g. compare a
heavily prefixing language to a heavily suffixing one on matched productive
morphology, or compare prefix vs suffix within a single language's *neologisms* to
watch lexicalization happen).

## Counterfactual hunt: is there a case where SUFFIXES generate more novel words?

Important logical point first: the serial-position hypothesis is about temporal
processing order, and morpheme time-order is universal in speech (prefix before
root, root before suffix). So the strict theory predicts NO concatenative spoken
language reverses the effect; a genuine reversal would falsify it. We cannot
propose a counterfactual the theory permits, only hunt for one.

The sharpest hunt is not a whole language but a morpheme CLASS, because it
dissociates serial-position from the main rival account. The rival: the asymmetry
is about grammatical FUNCTION, not position. English suffixes are class-changing
(verb -> noun, etc.), which forces regularity/transparency; English prefixes are
class-maintaining meaning-modifiers, free to drift. In English these are
confounded (the suffixes happen to be the class-changing ones). They come apart on
**class-maintaining suffixes**, above all diminutives/augmentatives
(Spanish bolso->bolsillo "pocket", manzana->manzanilla "chamomile",
ventana->ventanilla "ticket window", zapato->zapatilla "slipper").

Test (position held constant: both are suffixes; vary grammatical function):
Spanish and Italian, Numberbatch, vowel-restoration repair (bolsillo->bols->bolso).

```
language  diminutive suffixes   class-changing suffixes   p
Spanish   28.1% novel (n=481)   10.2% novel (n=1703)      3.8e-19
Italian   23.6% novel (n=1077)   8.3% novel (n=2398)      8.3e-38
```

Diminutive (class-maintaining) suffixes drift ~2.8x more than class-changing
suffixes, with position constant. They even exceed the prefix novelty rates in the
same languages (es prefix 20.6%, it 18.1%). So a SUFFIX subtype out-generates
prefixes. **This is the counterfactual.**

Interpretation: serial position alone is NOT sufficient. Grammatical function
(class-changing vs class-maintaining) is an independent, strong driver. A refined
hypothesis that fits everything so far: affixes that do NOT impose a regular
class-changing grammatical function bind holistically and drift. In English those
are exactly the prefixes (all class-maintaining); cross-linguistically they also
include evaluative suffixes. Serial position may still contribute (the LLM
left-to-right result suggests order matters), but it is one factor, not the whole
story. The honest model is two-factor: position AND function, confounded in
English, separable in the Romance diminutives.

Caveat: diminutive affixes (ito, ina, azo) are short and attract more orthographic
false splits than long class-changing affixes (cion, miento), and the most-novel
examples include junk (vagina = ina+vago). Some of the magnitude is artifact. But
the direction is large, significant in two languages, and matches a strong,
textbook linguistic prior (diminutives are classic lexicalization machines), so
the qualitative conclusion is safe; the exact magnitude is not.

Other candidate hunting grounds (not yet run), ranked by how decisive they would be:
1. **Class-maintaining suffixes more broadly** (the test above; could extend to
   Russian -ik/-ok, Portuguese -inho, German -chen/-lein diminutives).
2. **Agglutinative suffixing languages** (Turkish, Finnish, Hungarian, Japanese,
   Korean): the theory predicts their suffixal derivation stays transparent
   (root-first). Heavy suffix lexicalization there would strain it. Best big-data
   test; Numberbatch covers tr/fi/hu/ja/ko. No within-language prefix contrast.
3. **Right-to-left scripts** (Hebrew, Arabic): dissociate visual reading order
   from speech-time order. Theory says script direction should not matter
   (speech is still root-first for suffixes). Templatic morphology complicates.
4. **Bantu** (Swahili): derivational morphology is suffixal (verb extensions),
   inflection is prefixal; test whether the suffixal derivation lexicalizes.

## Test 6 result: agglutinative suffixing (Turkish, Finnish). INCONCLUSIVE.

Turkish and Finnish are heavily suffixing with almost no prefixes, so no
within-language prefix contrast. The serial-position account predicts their
suffixal derivation stays transparent (root is first = anchor).

```
language  class-changing suffixes   diminutive
Turkish   11.7% novel (n=1086)      no decompositions (cik undergoes consonant change)
Finnish   20.2% novel (n=1521)      11.0% (-nen, but mostly adjective marker, not a real diminutive)
```

Turkish leans transparent (11.7%, similar to the Romance class-changing level),
consistent with the prediction. Finnish came out noisier (20.2%) but inspection
shows that is mostly false splits from consonant gradation defeating the crude
free-root match (yllatys->ylla, kataja->kata). The diminutive split did not work:
Turkish -cik triggers consonant alternation so few free-root matches survived, and
Finnish -nen is an adjective marker more than a diminutive (examples were
vammainen, yleinen). Verdict: INCONCLUSIVE. A real agglutinative test needs proper
morphological analyzers (Zemberek for Turkish, Omorfi/Voikko for Finnish) to
separate derivation from inflection and handle vowel harmony / gradation. Turkish
is mildly supportive; Finnish is uninterpretable as run.

## Literature positioning. THE IDEA IS PARTLY FORESHADOWED (see LITERATURE.md).

Three-strand review. The honest finding: the components mostly exist; our novel
contributions are the synthesis and two specific results. Closest ancestors:
- **Cutler, Hawkins & Gilligan 1985, "The suffixing preference: a processing
  explanation."** Already argue word beginnings are the access cue and prefixes
  are costly *because they precede the stem* (serial position), explaining the
  typological suffixing preference. They do NOT derive a semantic-drift
  consequence. ~60% of our premise. Our move: convert it into a diachronic
  semantic-drift prediction about existing lexemes.
- **Lazaridou et al. 2013.** Owns the additive offset metric we use; the
  prefix<suffix compositionality asymmetry may already be foreshadowed there.
  (Must read directly to confirm.)
- **Stupak & Baayen 2022.** German particle verbs less transparent than
  suffixation, same direction as our headline, different construction.
- **Hay 2001.** Frequency drives holistic storage and opacity, but via *relative*
  base:derived frequency, not position. Note: Stupak & Baayen report POSITIVE
  frequency-transparency, opposite to our absolute-frequency finding; must
  reconcile (different frequency variable).
- **Jurafsky 1996.** Diminutive drift documented, but not via the class-changing
  vs class-maintaining contrast.

Genuinely novel (defensible): (1) the position-vs-function dissociation via
diminutives (strongest); (2) the autoregressive-LLM substrate-independence result
(needs a BERT control to rule out a pure distributional artifact); (3) breaking
diachronic change down by morphological type (unaddressed, but our result was
weak). NOT novel: the metric, the basic transparency measurement, and arguably the
bare prefix<suffix asymmetry.

**Highest-value next experiment (flagged by the review):** add a bidirectional
baseline (BERT) to the LLM test. If BERT shows a weaker/absent asymmetry, the
autoregressive-order interpretation is supported; if equal, it collapses to a
distributional account. This single control decides whether the LLM result is a
genuine processing-order finding or a restatement of Lazaridou 2013.

Full citations and the referee-objections list are in `LITERATURE.md`.

## Test 7 result: BERT control. NEGATIVE. Autoregressive interpretation REFUTED.

The decisive control the review demanded. Compared an autoregressive model (GPT-2)
with a bidirectional one (BERT), identical extraction (mean-pooled sub-token
last-layer hidden states, centered + top-1 PC removed), so architecture is the
only difference.

```
model               prefix_nov  suffix_nov  ratio   gap     p
gpt2 (autoregress.)    22.8%       14.1%     1.62  +0.134  7.0e-51
bert (bidirectional)   20.8%       13.2%     1.57  +0.118  9.8e-84
```

BERT shows the asymmetry essentially as strongly as GPT-2 (ratio 1.57 vs 1.62, a
noise-level difference; the tiny gap difference is not meaningfully attributable
to architecture given the confounds of different tokenizer/training/dims). So the
asymmetry is NOT a signature of left-to-right autoregressive processing. A
bidirectional model that never "drives through the prefix" reproduces it.

**Conclusion: the autoregressive / substrate-independence interpretation is
disconfirmed.** The prefix<suffix asymmetry is a DISTRIBUTIONAL property, baked
into corpus co-occurrence statistics (exactly why static embeddings, Lazaridou
2013, showed it too). An LM of either architecture simply reflects the already-
lexicalized state of the language. The embedding results across this whole project
(static, GPT-2, BERT) measure the OUTCOME (current non-compositionality), not the
MECHANISM of how words got that way.

What this does to the thesis:
- The robust empirical OUTCOME stands: prefixed words are less compositional than
  suffixed, cross-linguistically, and class-maintaining diminutive suffixes drift
  more than class-changing ones (the position-vs-function dissociation).
- The serial-position MECHANISM for human lexicalization is neither confirmed nor
  refuted by LMs (they only show the outcome). Its closest support remains the
  human-side processing argument (Cutler, Hawkins & Gilligan 1985). The diminutive
  dissociation actively favors grammatical FUNCTION over pure position.
- The flashy "a next-token model proves it is about processing order" claim is
  dead. Honest negative result; the control did its job.

The paper (PAPER.md) was revised to report this control and reframe the LLM
section and discussion accordingly: the LLM result is now "the asymmetry is a
distributional property present in both autoregressive and bidirectional LMs", not
evidence for processing order.

## Methodological correction (Kevin): bifurcate human vs LLM tracks

Do NOT lump the LLM results with the human results. Reason (sharper than first
stated): the LLMs are TRAINED ON HUMAN TEXT, so they are downstream of the human
data. An LLM reproducing the human asymmetry is the model learning the human
distribution, not independent corroboration; lumping them double-counts. Two
separate tracks:
- HUMAN track: facts about human language (the object of study). All the
  static-embedding, morphological, diachronic, cross-linguistic results.
- LLM track: how an LLM internally organizes morphology. Only independently
  interesting where it DIVERGES from the human distributional baseline. Our
  control found no divergence (GPT-2 ~ BERT), so the LLM track says only: LLMs
  inherit the human distribution; architecture does not matter. To show a genuine
  shared MECHANISM (not mere inheritance) you would need the LLM to do something
  the training distribution did not hand it, e.g. lexicalize NONCE affixed words
  the way humans would. Separate future test.

## Human-only reanalysis: simple function hypothesis FAILS; referentiality emerges

Clean within-English test (validated MorphoLEX, GloVe), three groups:
```
group                         mean_comp   novel<0.15
prefix (class-maintaining)      +0.265      24.6%
suffix, class-maintaining       +0.374      15.6%   <- MOST transparent, not least
suffix, class-changing          +0.302      21.0%
```
The simple FUNCTION hypothesis (class-changing vs class-maintaining) is REJECTED:
class-maintaining suffixes are the most transparent. The earlier "function beats
position" reading from the Romance diminutives over-generalized; I had conflated
"class-maintaining" with "diminutive". They are different (the English CM group is
dominated by regular, transparent -ism/-ist).

But the per-suffix gradient reveals a better variable, cutting across POS:
- transparent end: -ness(3.3%), -est(0.5%), -ism(4.3%), -ics(0%), -ity(11.5%),
  -acy, -ancy, -ion. These derive ABSTRACT qualities/states/degrees.
- drifty end: -er/-or (agent/instrument), -age, -ling, -ster, -ory(place),
  -ine/-ite/-ate (substances, e.g. chemical names), -oid (shape). These NAME
  CONCRETE things.

Hypothesis the human data SUGGEST: the driver of lexicalization is REFERENTIALITY /
CONCRETENESS, not affix position and not POS-change. A derived word that names a
specific thing in the world (agent, instrument, substance, place, diminutive
object) acquires real-world properties beyond its parts and drifts; a word for an
abstract quality has no referent to specialize and stays compositional. This
subsumes the Romance diminutives (bolsillo = a specific object) and reframes them.

Caveats: this is POST-HOC, read off a per-suffix table with some noisy/false-split
cells (-ar = dollar/grammar, -ite includes proper/mineral names). -ist names a
person yet stays transparent (artist = one who does art), so referentiality is not
perfectly clean either. NOT a confirmed result.

Clean confirmatory test to run next (no affix hand-coding): correlate each derived
word's NON-compositionality with its CONCRETENESS rating (Brysbaert et al. 2014,
40k English words, free). If concrete derived words drift more than abstract ones,
referentiality is supported on pure human data. This is the principled follow-up.
