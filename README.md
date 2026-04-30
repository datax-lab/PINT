# PINT: Pathway–Pathway Interaction Model

## Overview

Understanding disease mechanisms requires modeling the coordinated activity of multiple biological pathways rather than treating them as independent entities. Most existing pathway-based deep learning approaches aggregate pathway representations using fully connected layers, ignoring interactions between pathways. This limitation can restrict both predictive performance and biological interpretability.

## Motivation

Disease progression is driven by complex interactions across biological pathways. However, current models:

- Treat pathways independently  
- Ignore inter-pathway relationships  
- Miss biologically meaningful interactions  

To address this gap, we propose a model that explicitly learns **pathway–pathway interactions**.

## Method

We introduce **PINT (Pathway–Pathway Interaction Network)**, a deep learning framework that:

- Learns pathway representations from gene expression data  
- Models interactions between pathways using a **self-attention mechanism**  
- Uses an **attention-based pooling layer** to identify patient-specific pathway importance  

## Results

We evaluated PINT on five TCGA cancer datasets for survival analysis.

Key findings:

- **Consistently outperforms benchmark models**  
- Identifies **pathways significantly associated with survival**  
- Reveals **biologically meaningful pathway–pathway interactions**  

### Case Study: BRCA

On the BRCA dataset:

- PINT identified important survival-associated pathways  
- Discovered patient-specific pathway interactions  
- Many findings are supported by existing literature  

Notably:

- The **RAS signaling pathway** was strongly associated with survival  
- Learned interactions captured known relationships with:
  - cAMP signaling  
  - TNF signaling  
  - Rap1 signaling  

## Key Contributions

- Introduces a novel framework for **modeling pathway–pathway interactions**
- Improves both **predictive performance** and **interpretability**
- Provides **patient-specific biological insights**

---

## Repository Structure

```bash
data_preprocess/   # Contains TCGA-BRCA dataset and preprocessing scripts
exp_setup/         # Train/validation/test split generation
exp_runs/          # Model training scripts (Slurm + Python)
metadata/          # KEGG pathway information
