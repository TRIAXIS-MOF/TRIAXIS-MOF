# 🌌 TRIAXIS-MOF  
### Triple-Modality Analytics for eXplainable, Integrated Screening of Metal–Organic Frameworks

**TRIAXIS-MOF** is an integrated, data-driven screening framework designed to identify high-performance metal–organic frameworks (MOFs) for **industrial-scale hydrogen storage**.  
This platform unifies three analytical axes—**white-box machine learning**, **process-level energy & techno-economic evaluation**, and **DFT-based structural/electronic stability**—into a single **explainable decision pipeline**.

Built on a dataset of **98,694 CoREMOF structures**, TRIAXIS-MOF delivers transparent predictions, robust energy fingerprints, and physics-consistent quantum validation to accelerate the discovery of viable hydrogen-storage materials.

---

## ✨ Key Features

---

### 1. **White-Box Machine Learning (Interpretability-First)**  
- Polynomial / feature-transparent models  
- Correlation mapping: PLD, LCD, density vs. working capacity  
- 5-fold cross-validation across 90:10–50:50 splits  
- **High model fidelity**:
  - Gravimetric: `R² = 0.9957`, `RMSE = 0.2623`
  - Volumetric: `R² = 0.9476`, `RMSE = 2.44`

---

### 2. **Process-Level Energetics & MCDM Decision System**  
- Joback heat-capacity estimation (≤10% deviation)  
- Energy Fingerprint analysis: **1.6–5.4 MJ synthesis energy**  
- Five MCDM ranking methods:
  - WSM, Entropy, TOPSIS, VIKOR, PROMETHEE  
- Unified scoring using **Borda Count**  
- Identified top candidates: **FATQID, NAWXER, VOLPET, YAVWUQ, YUGLES**

---

### 3. **DFT-Based Electronic & Structural Verification**  
- Converged calculations:
  - 15–50 Ry cutoffs  
  - 1×1×1 to 5×5×5 k-point grids  
- Physical consistency:
  - Forces `< 0.01 Ry/Bohr`  
  - Total energies `−826 to −2043 Ry`  
  - Stable DOS features & minimal stress  
- **Cu–O d–p hybridization** analyzed for Kubas-type H₂ interactions

---

## 🚀 Outcome

TRIAXIS-MOF identifies **FATQID** as the leading candidate for industrial H₂ storage based on:

- **48 g/L** working capacity  
- Stable **Cu–O** electronic structure  
- Low synthesis cost: **24.6 USD/kg H₂**  
- High energy efficiency: **1.6–2.0 MJ**

This triple-modality pipeline provides a **transparent, reproducible, and scalable strategy** for MOF discovery—bridging **machine learning**, **process engineering**, and **quantum chemistry**.

---

## 🔑 Keywords  
Hydrogen storage • Metal–organic frameworks • White-box machine learning • Explainable AI • DFT • Energetics • MCDM • Materials screening


