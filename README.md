# Long-Duration Energy Storage — Window 1 Portfolio Modelling (2026)

Independent modelling of Ofgem’s Window 1 Long-Duration Energy Storage (LDES) portfolio using publicly available data. This project explores how physical duration, geographic constraint exposure, and technology-specific lifecycle costs shape project-level economics.

### Latest Update: Ofgem Consultation Addendum
In 2026, I submitted a formal consultation response and technical Addendum to Ofgem's Window 1 minded-to decision. This repository contains the underlying project-level modelling, data pipelines, and the full text of the Addendum supporting that submission.

**Formal Response:**

**[📄 Read the Response to Ofgem Consultation: Long-Duration Electricity Storage Cap and Floor Framework ]()**

**Addendum:**

**[📄 Read the Full Addendum: Compounding Physics and Regulatory Risks]()** 

---

### Overview
The modelling pipeline covers:
- Lifecycle cost curves (Li-ion augmentation, PSH/CAES refurbishment)
- Duration-constrained wholesale dispatch (daily vs. multi-day optimisation)
- Constraint-zone mapping and RRT (Revenue Requirement Target) persistence modelling
- Curtailment-adjusted LCOS calculations
- Revenue stack integration (wholesale, Capacity Market, Balancing Mechanism)
- Technology-specific risk and financing adjustments

The analysis is exploratory and intended to support transparent, evidence-based understanding of how duration, location, and lifecycle economics interact within the Window 1 portfolio.

### Key Patterns Observed
- **Duration is the primary driver:** Across Li-ion assets, duration remains the strongest predictor of residual support requirements.
- **The Li-ion "Fixed-Cost Squeeze":** Constrained assets (e.g., behind the SSEN-S boundary) face a compounding penalty. Physical curtailment (the bottleneck) shrinks the dispatch revenue base, while fixed TNUoS capacity charges (the tollgate) remain static. A national duration curve is insufficient; constrained assets require an explicit locational constraint coefficient.
- **The PSH Financing Wedge:** Under central market assumptions, PSH shows near-zero modelled residual floor requirements. However, forcing 60–80 year civil infrastructure through a 25-year private financing framework creates an estimated **£245m/yr** avoidable risk premium across the Window 1 PSH portfolio. This highlights a structural mismatch that requires differentiated calibration and complementary public financing tools (e.g., via HMT/DESNZ) to prevent consumers from paying a deadweight cost for policy uncertainty.

### Contents
- `/notebooks/` — Jupyter notebooks for dispatch modelling, LCOS, and regression analysis.
- `/data/processed/` — Publicly sourced datasets (Ofgem Window 1 annexes, ELEXON BMRS price and constraint data).
- `/docs/` — Consultation response documents and the technical Addendum. *(Add this folder if you upload the PDF/Markdown here)*

### Reproducibility
All modelling is based solely on public data and reproducible methods. The goal is to provide a transparent, open-source baseline that the industry can review, challenge, and build upon.

### Status
This project actively supports a consultation response submitted to Ofgem (2026). Further updates may follow as additional data, final decisions, or future consultation windows become available.
