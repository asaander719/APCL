
# APCL: Adaptive Preference Modeling with Contrastive Learning for Fashion Matching


Official implementation of **APCL (Adaptive Preference modeling with Contrastive Learning)**, a novel framework for personalized fashion complementary recommendation that integrates multi-modal data, indirect relationships, and contrastive learning.

## 📖 Abstract

Fashion complementary recommendation faces unique challenges: personalization needs, multi-modal data complexity, and severe data sparsity. APCL addresses these by:

- **Adaptive Preference Modules**: Mining implicit user-product relationships through attention-based transformers
- **Dual Contrastive Learning**: Aligning functional views (Personal vs Indirect Personal, Compatibility vs Indirect Compatibility) across modalities
- **Multi-modal Integration**: Leveraging both visual and textual features for comprehensive compatibility modeling

## 🚀 Features

- **Multi-modal Fusion**: Integrates visual and textual features using pre-trained CLIP encoders
- **Adaptive Preference Modeling**: Captures indirect user-product relationships to alleviate data sparsity
- **Dual Contrastive Loss**: Aligns direct and indirect representations across different functional views
- **BPR Integration**: Combines Bayesian Personalized Ranking with contrastive learning for optimized recommendations


## 🏗️ Model Architecture

The APCL framework consists of four main components:
1. **Personal Preference Module (P)**: Models user-specific preferences
2. **Product Compatibility Module (C)**: Learns item-item compatibility
3. **Indirect Personal Preference (IP)**: Captures implicit user relationships
4. **Indirect Compatibility (IC)**: Models implicit item compatibility

## ⚙️ Datasets
- Download the two public datasets we use in the paper at:
  https://drive.google.com/file/d/1Dg7918zUGcL7tzs_OisNzc_FxYlQMG4E/view?usp=sharing

- Unzip the datasets and move them to **./dataset/**

## 📥 Training

`python APCL/run_APCL_Polyvore_RB.py`