# Long‑Duration Energy Storage — Window 1 Portfolio Modelling (2026)

This project contains independent modelling of Ofgem’s Window 1 Long‑Duration Energy Storage (LDES) portfolio using publicly available data. The aim is to explore how duration, location, and technology lifecycle costs shape project‑level economics.

**Overview**

The modelling pipeline covers:

- Lifecycle cost curves (Li‑ion augmentation, PSH/CAES refurbishment)

- Duration‑constrained wholesale dispatch (daily vs multi‑day optimisation)

- Constraint‑zone mapping and RRT persistence modelling

- Curtailment‑adjusted LCOS calculations

- Revenue stack integration (wholesale, CM, BM)

- Technology‑specific risk adjustments

This is exploratory analysis intended to support understanding of how physical duration and geographic constraints interact with lifecycle economics.

**Key Patterns Observed**

Duration shows a strong, linear influence on residual support needs across Li‑ion assets.

Constraint‑zone projects behind the SSEN‑S interface exhibit materially higher curtailment‑adjusted costs.

PSH assets appear structurally resilient, with near‑zero residual floor requirements under central and IDC‑stress assumptions.

**Contents**

/notebooks/ — Jupyter notebooks for dispatch modelling, LCOS, and regression analysis

/data/ — Publicly sourced datasets (Ofgem Window 1, BMRS price data)

/figures/ — Regression plots, duration curves, constraint‑zone maps

/analysis/ — Python modules for lifecycle economics and curtailment modelling

/docs/ — Summary notes and methodology references

**Status**

This project supports a consultation response submitted to Ofgem (2026).
All modelling is based solely on public data and reproducible methods.
