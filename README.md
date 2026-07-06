# Long‑Duration Energy Storage — Window 1 Portfolio Modelling (2026)

Independent modelling of Ofgem’s Window 1 Long‑Duration Energy Storage (LDES) portfolio using publicly available data. The project explores how physical duration, geographic constraint exposure, and technology‑specific lifecycle costs shape project‑level economics.

**Overview**

The modelling pipeline covers:

- Lifecycle cost curves (Li‑ion augmentation, PSH/CAES refurbishment)

- Duration‑constrained wholesale dispatch (daily vs multi‑day optimisation)

- Constraint‑zone mapping and RRT persistence modelling

- Curtailment‑adjusted LCOS calculations

- Revenue stack integration (wholesale, CM, BM)

- Technology‑specific risk adjustments

The analysis is exploratory and intended to support understanding of how duration, location, and lifecycle economics interact within the Window 1 portfolio.

**Key Patterns Observed**

- **Duration** is the strongest driver of residual support requirements across Li‑ion assets.

- **Constraint‑zone exposure (SSEN‑S)** materially increases curtailment‑adjusted costs.

- **PSH** assets appear structurally resilient, with near‑zero residual floor requirements under central and IDC‑stress assumptions.

**Contents**

/notebooks/ — Jupyter notebooks for dispatch modelling, LCOS, and regression analysis

/data/processed — Publicly sourced datasets (Ofgem Window 1, BMRS price data)

**Reproducibility**

All modelling is based solely on public data and reproducible methods.

**Status**

This project supports a consultation response submitted to Ofgem (2026).

Further updates may follow as additional data or consultation windows become available.
