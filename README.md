# Detecting Moral Schemas in Interview Transcripts with Large Language Models

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands are run from the repository root with `python src/bin_morality_analysis.py <command>`;
`python src/bin_morality_analysis.py <command> --help` lists the options of each command.

## Data

`data/cache/morality.csv` contains, for every respondent (`Survey Id`), the gold-standard and coder labels, the
crowdworker labels, the binary outputs of every model and prompt/input variant, and the survey variables used in
the regressions. All figures and tables below are computed from this file.

The interview transcripts cannot be shared. Annotating interviews requires either the raw transcripts
(`data/interviews/waves/wave_<n>/*.txt`) or a local dump of the morality excerpts
(`data/interviews/misc/morality_texts.csv`, columns `Survey Id`, `Wave`, `Interview Code`, `Morality Text`,
`Morality Response`, `Morality Summary`). Neither is part of the repository.

**Validation design.** All model-development choices (input format, prompt variant, guided LDA fit and scaling)
are made on Wave 1 (development, `DEVELOPMENT_WAVES` in `src/helpers.py`). The selected configuration of each
method is then frozen and evaluated once on Wave 3 (held-out test, `TEST_WAVES`). The dictionary and the seed words
of the guided LDA (`MORALITY_VOCAB`, Table A1) contain only the category names and a few synonyms.

## Annotating interviews with a model

```bash
# Dictionary, guided LDA, SBERT and NLI on the three inputs (full excerpt, respondent answers, summary)
python src/bin_morality_analysis.py annotate --model wc        --input "Morality Text" "Morality Response" "Morality Summary"
python src/bin_morality_analysis.py annotate --model lda       --input "Morality Text" "Morality Response" "Morality Summary"
python src/bin_morality_analysis.py annotate --model sbert     --input "Morality Text" "Morality Response" "Morality Summary"
python src/bin_morality_analysis.py annotate --model nli_quant --input "Morality Text" "Morality Response" "Morality Summary"

# LLMs (API keys in OPENAI_API_KEY / OPENROUTER_API_KEY)
python src/bin_morality_analysis.py annotate --model deepseek_bin
python src/bin_morality_analysis.py annotate --model chatgpt_bin

# Use the local text dump instead of the raw transcripts; the outputs are also written into data/cache/morality.csv
python src/bin_morality_analysis.py annotate --model wc --source dump

# A new wave and a newer LLM, evaluated against gold labels
# (CSV with 'Interview Code', 'Wave', 'Intuitive', 'Consequentialist', 'Social', 'Theistic')
python src/bin_morality_analysis.py annotate --model gpt5mini_bin --waves 4 --gold data/interviews/misc/gold_wave_4.csv

# Only models that run locally (no transcript sent to a third party)
python src/bin_morality_analysis.py annotate --model qwen_local_bin --local-only
```

Outputs are written to `data/cache/morality_model-<model>.pkl`. Every LLM call is logged in
`data/cache/llm_logs/<model>.jsonl` (exact model returned by the provider, token usage, timestamp); interrupted runs
resume from the log. New LLMs are added to `LLM_REGISTRY` and new waves to `WAVE_MORALITY_QUESTIONS`
(both in `src/helpers.py`).

Crowd labeling of a wave (the CloudResearch task input, then the collected labels):

```bash
python src/bin_morality_analysis.py crowd --wave 3
python src/bin_morality_analysis.py crowd --wave 3 --labels data/interviews/misc/crowd_labeling_wave_3.csv
```

## Reproducing the figures and tables

Figures are saved to `data/plots/`, tables to `data/tables/`.

| Paper | Content | Command |
|---|---|---|
| Figure 1 | Input engineering for the baseline models (development, Wave 1) | `python src/bin_morality_analysis.py figure 1` |
| Figure 2 | Prompt engineering for the LLMs (development, Wave 1) | `python src/bin_morality_analysis.py figure 2` |
| Section 4.1.2 (text) | LLM input variants and GPT-3.5 (development, Wave 1) | `python src/bin_morality_analysis.py table llm-inputs` |
| Figure 3 | Frozen models and crowdworkers (test, Wave 3) | `python src/bin_morality_analysis.py figure 3` |
| Table 1 | Elements of the LLM prompt (the prompt is `llm_prompt` in `src/helpers.py`) | – |
| Table 2 | Performance by morality type (test, Wave 3) | `python src/bin_morality_analysis.py table 2` |
| Table 3 | Dictionary false positives, theistic morality (needs the text dump) | `python src/bin_morality_analysis.py table 3` |
| Table 4 | Dictionary false negatives, consequentialist morality (needs the text dump) | `python src/bin_morality_analysis.py table 4` |
| Figure 4 | Speaker-role manipulation (test, Wave 3) | `python src/bin_morality_analysis.py figure 4` |
| Table 5 | Regressing future action on moral schemas, DeepSeek V3 | `python src/bin_morality_analysis.py table 5` |
| Section 3.2 | Inter-coder reliability and adjudication rates | `python src/bin_morality_analysis.py table reliability` |
| Section 4.2 | Agreement of the models with each trained coder | `python src/bin_morality_analysis.py table coders` |
| Table A1 | Dictionary vocabulary / LDA seed words | `python src/bin_morality_analysis.py table A1` |
| Appendix 8.4 | Summarization prompt (`CHATGPT_SUMMARY_PROMPT` in `src/helpers.py`) | – |
| Table A2 | Descriptives of the future-behavior variables | `python src/bin_morality_analysis.py table A2` |
| Table A3 | Descriptives of the control variables | `python src/bin_morality_analysis.py table A3` |
| Table A4 | Regressions with network controls (DeepSeek V3) | `python src/bin_morality_analysis.py table A4` |
| Table A5 | Regressions with religion controls (DeepSeek V3) | `python src/bin_morality_analysis.py table A5` |
| Table A6 | Regressions with demographic controls (DeepSeek V3) | `python src/bin_morality_analysis.py table A6` |
| Table A7 | Regressions with all controls (DeepSeek V3) | `python src/bin_morality_analysis.py table A7` |
| Table A8 | Full regression table (DeepSeek V3) | `python src/bin_morality_analysis.py table A8` |
| Table A9 | Full regression table (GPT-4o-mini) | `python src/bin_morality_analysis.py table A9` |

Any figure or table can be computed on other waves with `--waves`, e.g. Figure 3 on the development wave (which
includes the crowdworkers of Wave 1): `python src/bin_morality_analysis.py figure 3 --waves "Wave 1"`, saved as
`data/plots/figure_3_wave_1.png` (outputs for non-default waves get the waves as suffix).
Tables 3 and 4 contain interview excerpts and must not be shared.
