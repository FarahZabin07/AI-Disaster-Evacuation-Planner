# AI-Based Disaster Evacuation Planner

An AI-based earthquake evacuation planning system that predicts road blockage risks, identifies safer evacuation routes, and allocates evacuees to shelters based on capacity and accessibility constraints.

## Overview

The system models a simplified road network of Dhaka, Bangladesh, and integrates machine learning and AI-based algorithms to generate adaptive evacuation plans under simulated earthquake scenarios.

The project combines:

* **Logistic Regression** — predicts the probability of road blockage.
* **Graph Search & A*** — identifies safe and efficient evacuation routes using a risk-aware road network.
* **Constraint Satisfaction Problem (CSP)** — assigns evacuation zones to shelters while respecting capacity and access constraints.
* **Simulation & Visualization** — evaluates and presents evacuation plans under different earthquake scenarios.

### System Pipeline

```text
Earthquake Scenario
        ↓
Data Processing
        ↓
Road Risk Prediction
        ↓
Graph Update
        ↓
A* Route Planning
        ↓
CSP Shelter Allocation
        ↓
Final Evacuation Plan
```

## Key Features

* Earthquake scenario simulation
* Road blockage risk prediction
* Risk-aware graph construction
* A* based evacuation routing
* Shelter capacity management
* Constraint-based evacuation allocation
* Route and scenario visualization
* Performance and validity evaluation

## Technology Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NetworkX**
* **Matplotlib**
* **Git & GitHub**

## Project Structure

```text
AI-Disaster-Evacuation-Planner/
│
├── data/
├── models/
├── graph/
├── routing/
├── allocation/
├── simulation/
├── visualization/
├── tests/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Evaluation

The system is evaluated across individual components and the complete evacuation pipeline, including:

* Logistic Regression prediction performance
* A* route cost and computation time
* CSP constraint satisfaction and shelter capacity
* End-to-end evacuation plan validity under simulated road closures

