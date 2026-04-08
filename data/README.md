# SciPredict Data Structure and Format

This document details the schema and organization of the SciPredict benchmark dataset. The data is organized into three primary CSV files located in the `data/` directory.

The benchmark comprises 405 expert-curated questions derived from empirical studies published after March 31, 2025. It covers three main scientific domains (Physics, Biology, Chemistry) and spans 33 specialized sub-fields.

## File Overview

1.  **`data/main_ds.csv`**: The core dataset containing prediction questions, details of experimental setup and measurements taken, background knowledge information, ground truth answers (GTAs), etc.
2.  **`data/human_baseline.csv`**: Responses and calibration metrics from human domain expert.
3.  **`data/rubrics.csv`**: Expert-curated evaluation criteria used to judge and score responses to free-form (FF) questions.

---

## 1. Main Dataset (`data/main_ds.csv`)

This file contains the inputs required for the models (experimental setup, measurements) and the targets for evaluation (ground truth).

| Column Name | Description                                                                                                                                               |
| :--- |:----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TASK` | A unique hex identifier for the task. Links across all three CSV files.                                                                                   |
| `DOMAIN` | The high-level scientific domain: `Physics`, `Biology`, or `Chemistry`.                                                                                   |
| `FIELD` | The specific sub-specialization (e.g., `Experimental Condensed Matter Physics`, `Neurobiology`).                                                          |
| `PQ_FORMAT` | The format of the prediction question. One of: `MCQ` (Multiple-Choice question), `Free-Format Question`, or `Numerical Values`.                           |
| `TITLE` | Title of the source research paper.                                                                                                                       |
| `URL` | Link to the original publication (arXiv, bioRxiv, ChemRxiv, etc.).                                                                                        |
| `PUBLISHING_DATE` | Date the paper first appeared online.                                                                                                                     |
| `EXPERIMENTAL_SETUP` | **[Input]** A detailed description of the system, conditions, and interventions.                                                                          |
| `MEASUREMENT_TAKEN` | **[Input]** Specific metrics recorded during the physical experiment.                                                                                     |
| `OUTCOME_PREDICTION_QUESTION` | **[Input]** The main query (question) asking the model to predict the result of an experiment.                                                            |
| `GTA` | **[Target]** The raw expert-curated Ground Truth Answer extracted from the paper.                                                                         |
| `CLEAN_GTA` | **[Target]** A formatted version of the `GTA` (e.g., the specific letter for MCQs or a cleaned string).                                                   |
| `REQUIRED_BACKGROUND_KNOWLEDGE` | **[Context]** Expert-curated context providing relevant scientific principles or prior findings. Used for the **BK** experimental condition in the paper. |
| `REQUIRED_BACKGROUND_KNOWLEDGE_HASHED` | The background knowledge split into individual hashed items with hashes used as references to BK lines.                                                   |
| `RELATED_PAPERS_DATA` | JSON metadata about cited papers or sources used to construct the background knowledge.                                                                   |

### Question Formats (`PQ_FORMAT`)
*   **MCQ:** The model must select the correct option (A, B, C, or D). `CLEAN_GTA` is one <u>or more</u> letters. 
*   **Free-Format:** The model must generate a textual explanation. Evaluated using the criteria in `rubrics.csv`. `GTA` used as final ground truth answer.
*   **Numerical Value:** The model must predict a value or range. `CLEAN_GTA` is typically a range (e.g., `[7.2-8.8]`).

---

## 2. Human Baseline (`data/human_baseline.csv`)

This file contains performance data from human domain experts. It captures their predictions under two conditions: **No Background Knowledge (NBK)** and **With Background Knowledge (BK)**, along with calibration metrics.

| Column Name | Description |
| :--- | :--- |
| `TASK` | Foreign key linking to `main_ds.csv`. |
| `PREVIOUS_KNOWLEDGE` | Boolean indicating if the expert had prior knowledge of the specific paper (to filter for data contamination). |
| `FIRST_ANSWER` | The expert's prediction under the **NBK** condition (Experimental Setup + Measurements only). |
| `FIRST_REASONING` | The expert's rationale for their NBK prediction. |
| `FIRST_CONFIDENCE` | Self-reported confidence score (1-5 scale) for the NBK prediction. |
| `FIRST_DIFFICULTY_SCORE` | Self-reported difficulty score (1-5 scale) for the NBK prediction. |
| `REVISED_ANSWER` | The expert's prediction under the **BK** condition (after receiving `REQUIRED_BACKGROUND_KNOWLEDGE`). |
| `REVISED_REASONING` | The expert's rationale after seeing the background knowledge. |
| `REVISED_CONFIDENCE` | Self-reported confidence score (1-5 scale) for the BK prediction. |
| `REVISED_DIFFICULTY` | Self-reported difficulty score (1-5 scale) for the BK prediction. |
| `FEASIBILITY_SCORE` | A 1-5 rating indicating how feasible it is to predict the outcome theoretically without running the physical experiment. |
| `FEASIBILITY_JUSTIFICATION` | The expert's explanation for their feasibility rating. |

---

## 3. Rubrics (`data/rubrics.csv`)

This file is used exclusively for evaluating **Free-Format** questions. It provides the structured criteria that a judge (e.g., the LLM judge) would use to determine if a provided answer fits the criteria of an accurate and complete answer given the `GTA`.

| Column Name | Description                                                                                                                                                                                                 |
| :--- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TASK` | Foreign key linking to `main_ds.csv`.                                                                                                                                                                       |
| `CRITERION_1` ... `CRITERION_10` | Individual evaluation criteria. Each criterion represents a specific fact or logical step that must be present in the model's answer to be considered correct and complete. Not all tasks use all 10 slots. |
| `NO_CRITERIA` | The integer count of active criteria columns for this specific task (>1).                                                                                                                                   |

**Evaluation Logic:**
For a Free-Format question to be marked as correct, the judge must verify that the provided response satisfies ALL the specific criteria listed in this file.
