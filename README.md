# Facial-based Heterogeneous Graph Representation Learning for Autism Spectrum Disorder Detection

This repository contains the implementation of **HGMoE**, a novel graph-based framework for early identification of children with Autism Spectrum Disorder (ASD) using facial expression dynamics. The model leverages heterogeneous graph construction, category-specific edge attention, and a Mixture-of-Experts mechanism to effectively model subtle behavioral patterns from videos.

## 🧠 Motivation

Children with ASD often exhibit atypical facial expressions and reduced coordination in non-verbal social cues. This project builds a **heterogeneous graph representation** from facial features over time and models **temporal, semantic, and action-unit-based relationships** between facial states. The goal is to improve ASD detection in small-sample, dynamic, and multi-dimensional scenarios.

---

## 🌐 Overview of HGMoE Framework

The HGMoE pipeline involves:

1. **Face Feature Extraction**: Extract 1000-D features using ResNet50 from 30 evenly sampled frames per video.
2. **Graph Construction**:
   - Nodes represent frames (facial states).
   - Edges are constructed based on:
     - **Temporal** adjacency.
     - **Semantic** similarity (cosine distance > θ).
     - **AU co-activation** patterns.
3. **Heterogeneous Graph Learning**:
   - Incorporates edge **category-aware attention**.
   - Leverages a **Mixture-of-Experts (MoE)** module to capture diverse relations.
   - Multi-loss training with expert diversity and attention regularization.
