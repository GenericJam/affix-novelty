# Literature positioning and novelty assessment

Honest answer to "is this novel, and what is the closest prior work?" Based on a
three-strand review (distributional morphology, psycholinguistics / historical
linguistics, and NLP / LLMs). Short version: **the components mostly exist; the
synthesis and two specific contrasts are the novel part.** This is a
"new-synthesis-plus-targeted-new-results" project, not a from-scratch discovery.

## The two closest ancestors (read these first)

1. **Cutler, Hawkins & Gilligan (1985), "The suffixing preference: a processing
   explanation," *Linguistics* 23(5):723-758.** This is the closest conceptual
   ancestor of the serial-position hypothesis. They already argue that word
   beginnings are the privileged access cue and that **prefixes are costly
   precisely because they sit before the stem** (a genuine serial-position
   argument), which explains the cross-linguistic suffixing preference. What they
   do NOT do: derive a *semantic-drift / lexicalization* consequence for existing
   prefixed words. They explain why languages prefer to *build* suffixed words;
   we predict that existing *prefixed* words *drift in meaning more*. Converting
   their processing/typology argument into a diachronic semantic claim is our move.
   (Follow-up: Hawkins & Cutler 1988, "Psycholinguistic factors in morphological
   asymmetry," in *Explaining Language Universals*, brings transparency into the
   same frame; full chapter text was not verified in review and should be read
   directly, as it is the source most likely to already contain a
   transparency-by-position remark.)

2. **Lazaridou, Marelli, Zamparelli & Baroni (2013), "Compositionally Derived
   Representations of Morphologically Complex Words in Distributional Semantics,"
   ACL 2013.** This owns the **additive vector-offset method** we use
   (derived ≈ base + affix offset) and analyzes error by affix type; the review
   indicates suffixed derivatives come out more compositionally predictable than
   prefixed ones. Our compositionality metric is essentially their additive model
   repurposed as a per-word transparency probe. **Implication: our metric is not
   novel, and the prefix<suffix asymmetry may already be foreshadowed here. Read
   this paper directly to confirm exactly what they reported by affix type before
   making any novelty claim about the asymmetry itself.**

## Established (NOT novel) in our project

- **The additive offset metric.** Lazaridou et al. 2013; applied in analogy form
  by Gladkova, Drozd & Matsuoka 2016 (BATS), which even separates prefix/suffix
  categories.
- **Measuring derivational semantic transparency distributionally** (composed vs
  observed vector). Marelli & Baroni 2015 (Psych Review, affix-as-matrix, with a
  validated transparency measure); Padó, Herbelot, Kisselew & Šnajder 2016
  (COLING, which factors predict transparency in German); Kisselew et al. 2015.
- **A prefix-like vs suffix transparency asymmetry in our direction already
  exists.** Stupak & Baayen 2022 (*The Mental Lexicon*) find German separable
  particle verbs less compositional/transparent than suffixation. This is the
  single biggest threat to novelty claim #1, though particle verbs are not
  English derivational prefixes.
- **Synthesizing meaning from morphemes including bound pieces.** Cotterell &
  Schütze 2018 (TACL) jointly segment and synthesize derived-word meaning;
  closest to our bound-root (cranberry) reconstruction, though they do not isolate
  cranberry roots like *-ceive*.
- **Frequency drives holistic storage and opacity.** Hay 2001 (*Linguistics*),
  Hay & Baayen 2005 (*TiCS*): when the derived form outweighs its base, the word
  is stored whole and parsability/transparency drop. This is the closest existing
  *mechanism* for our conclusion, but the driver is **relative** base:derived
  frequency, not serial position.
- **Diminutive semantic drift.** Jurafsky 1996 (*Language*) models the diminutive
  as a radial polysemy category. Diminutive lexicalization is textbook.
- **Anisotropy and its correction.** Ethayarajh 2019 (measured GPT-2 anisotropy
  specifically); Mu & Viswanath 2018 (all-but-the-top); Gao et al. 2019
  (representation degeneration). Exactly the correction we apply.
- **Diachronic embeddings and laws of semantic change.** Hamilton, Leskovec &
  Jurafsky 2016 (HistWords; laws of conformity and innovation, i.e. frequency and
  polysemy). SemEval-2020 Task 1 (Schlechtweg et al.).

## Appears novel (defensible, with caveats)

1. **The position-vs-function dissociation via class-maintaining diminutive
   suffixes** (Spanish/Italian -illo/-ito drift MORE than class-changing -ción).
   No prior distributional work on Romance evaluative-suffix transparency of this
   kind was found. This is the strongest novel contribution. Jurafsky documents
   diminutive drift but never via a class-changing vs class-maintaining contrast,
   and the use of that contrast as a transparency modulator appears unclaimed.
2. **Demonstrating the prefix<suffix asymmetry inside an autoregressive LLM
   (GPT-2 hidden states) and tying it to left-to-right processing order
   (substrate-independence).** The asymmetry in *static* embeddings is older
   (Lazaridou 2013); showing it in a contextual autoregressive model and giving
   it a processing-order interpretation is new. Closest directional-processing
   precedent: Ács et al. 2023/2024 (mBERT, left context ~40% more informative than
   right) but that is bidirectional, about context not morpheme order, and not
   tied to autoregressive training.
3. **Breaking diachronic semantic change down by prefix vs suffix.** No
   HistWords-style study stratifies semantic change by morphological type. Genuinely
   unaddressed, though our own diachronic result was weak (the asymmetry is old
   and stable, not accelerating in the modern window).
4. **A systematic English prefix-vs-suffix non-compositionality contrast on
   validated morphology (MorphoLEX) across multiple embeddings, including
   bound-root families.** Foreshadowed by Lazaridou 2013 and Stupak & Baayen 2022;
   our value-add is scale, validation, and the bound-root reconstruction.

## The three things a referee will press, and what to do

1. **"This is just the suffixing-preference processing account (CHG 1985)."**
   Response: we predict a *semantic/diachronic* outcome (existing prefixed words
   are less compositional) that CHG never derive; foreground this and cite them
   prominently. Read Hawkins & Cutler 1988 in full first.
2. **"Your frequency direction contradicts the literature."** Stupak & Baayen 2022
   report *positive* frequency-transparency; Hay 2001 ties opacity to *relative*
   frequency. We report opacity rising with *absolute* word frequency. These are
   not the same variable (absolute vs relative base:derived), but we must
   reconcile this explicitly or it reads as an artifact.
3. **"GPT-2 just inherits a distributional artifact; the autoregressive-order
   story is unsupported."** Because the asymmetry exists in static (non-directional)
   embeddings already, the left-to-right interpretation needs a control. **Add a
   bidirectional baseline (BERT): if BERT shows a weaker/absent asymmetry, the
   autoregressive-order claim is supported; if BERT shows it equally, the
   interpretation collapses to a pure distributional account.** This is the single
   highest-value next experiment.

## Bottom line

The honest framing for a write-up: we did not discover that prefixed words are
less compositional than suffixed words; that is foreshadowed in distributional
semantics (Lazaridou 2013) and in a processing/typology form in psycholinguistics
(Cutler, Hawkins & Gilligan 1985). Our contributions are (a) the
position-vs-function dissociation via diminutives, (b) the autoregressive-LLM
substrate-independence result (pending a BERT control), and (c) a unified
serial-position + grammatical-function account that ties these literatures
together and makes the diachronic prediction explicit. Claim those, cite the
ancestors generously, and run the BERT control before calling the LLM result a
finding.

## Key references

- Cutler, A., Hawkins, J. A., & Gilligan, G. (1985). The suffixing preference: a processing explanation. *Linguistics* 23(5), 723-758.
- Hawkins, J. A., & Cutler, A. (1988). Psycholinguistic factors in morphological asymmetry. In J. A. Hawkins (ed.), *Explaining Language Universals*. Blackwell.
- Lazaridou, A., Marelli, M., Zamparelli, R., & Baroni, M. (2013). Compositionally derived representations of morphologically complex words in distributional semantics. *ACL 2013*.
- Marelli, M., & Baroni, M. (2015). Affixation in semantic space: modeling morpheme meanings with compositional distributional semantics. *Psychological Review* 122(3), 485-515.
- Padó, S., Herbelot, A., Kisselew, M., & Šnajder, J. (2016). Predictability of distributional semantics in derivational word formation. *COLING 2016*.
- Kisselew, M., Padó, S., Palmer, A., & Šnajder, J. (2015). Obtaining a better understanding of distributional models of German derivational morphology. *IWCS 2015*.
- Stupak, I., & Baayen, R. H. (2022). An inquiry into the semantic transparency and productivity of German particle verbs and derivational affixation. *The Mental Lexicon* 17(3), 422-457.
- Cotterell, R., & Schütze, H. (2018). Joint semantic synthesis and morphological analysis of the derived word. *TACL* 6.
- Gladkova, A., Drozd, A., & Matsuoka, S. (2016). Analogy-based detection of morphological and semantic relations with word embeddings. *NAACL-HLT SRW 2016* (BATS).
- Hay, J. (2001). Lexical frequency in morphology: is everything relative? *Linguistics* 39(6), 1041-1070.
- Hay, J., & Baayen, R. H. (2005). Shifting paradigms: gradient structure in morphology. *Trends in Cognitive Sciences* 9(7), 342-348.
- Jurafsky, D. (1996). Universal tendencies in the semantics of the diminutive. *Language* 72(3), 533-578.
- Taft, M., & Forster, K. I. (1975). Lexical storage and retrieval of prefixed words. *JVLVB* 14(6), 638-647.
- Marslen-Wilson, W., & Welsh, A. (1978). Processing interactions and lexical access during word recognition in continuous speech. *Cognitive Psychology* 10(1), 29-63.
- Hofmann, V., Pierrehumbert, J., & Schütze, H. (2021). Superbizarre is not superb: derivational morphology improves BERT's interpretation of complex words. *ACL-IJCNLP 2021*.
- Hofmann, V., Pierrehumbert, J., & Schütze, H. (2020). DagoBERT: generating derivational morphology with a pretrained language model. *EMNLP 2020*.
- Edmiston, D. (2020). A systematic analysis of morphological content in BERT models for multiple languages. *arXiv:2004.03032*.
- Ács, J., et al. (2023/2024). Morphosyntactic probing of multilingual BERT models. *Natural Language Engineering* 30 (arXiv:2306.06205).
- Ethayarajh, K. (2019). How contextual are contextualized word representations? *EMNLP 2019*.
- Mu, J., & Viswanath, P. (2018). All-but-the-top: simple and effective postprocessing for word representations. *ICLR 2018*.
- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. *ACL 2016*.
- Schlechtweg, D., et al. (2020). SemEval-2020 Task 1: unsupervised lexical semantic change detection. *SemEval 2020*.
