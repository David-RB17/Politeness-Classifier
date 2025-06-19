# Notes on Label Format Change (Politeness Classification)

## Context

Originally, the dataset used for the Japanese politeness classification project had **three labels**:

- `0` — Informal / Casual (ため口)
- `1` — Neutral (普通)
- `2` — Formal / Polite (敬語)

Several EDA notebooks in this folder were created based on this **three-class setup**, including visualizations and class distribution analyses.

---

## Label Simplification

To streamline the classification task and improve model performance on limited data, we decided to **merge the labels into a binary format**:

- `0` — Informal (combines previous 0 and 1: ため口 + 普通)
- `1` — Formal (same as previous 2: 敬語)

This shift reflects a more practical focus on distinguishing **casual vs. formal tone**, rather than capturing subtle in-between cases like "neutral" speech.

---

## Why This Change?

- **Better separation** between categories (neutral speech often overlaps with casual/formal)
- **Improved training stability** with fewer classes and clearer signal

---

## Note

If you are reviewing older notebooks:
- Assume the label meanings are still in the **original 0/1/2 format**
- Any new notebooks or scripts will now use the **binary 0/1 format**

A script to perform this label conversion is available in the root directory: `convert_labels.py`.

---

_Last updated: June 2025_
