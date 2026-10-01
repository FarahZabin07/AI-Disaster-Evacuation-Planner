# AI-Based Earthquake Disaster Evacuation Planner


> An AI-powered, risk-aware evacuation planning system for Dhaka that combines earthquake data, machine learning, geospatial road networks, constraint satisfaction, and A* pathfinding to recommend safer evacuation routes and feasible emergency shelters.

---

## Overview

Dhaka is highly vulnerable to earthquake-related disruption due to its dense population, complex road network, and proximity to regional seismic sources such as the **Madhupur** and **Dauki** fault systems.

Conventional navigation systems primarily optimize for distance or normal traffic conditions. They do not inherently account for potential post-earthquake road blockage or network disruption.

This project addresses the problem by transforming earthquake information into **road-level risk estimates** and using those predictions to dynamically modify a Dhaka road network before performing evacuation planning.

The system combines:

**USGS Earthquake Data → Logistic Regression → Risk-Aware Road Graph → CSP Shelter Selection → A* Route Planning**

---

## Key Features

* Real-time and historical earthquake data ingestion through the USGS API
* Dhaka road-network extraction using OpenStreetMap and OSMnx
* Earthquake-to-road geospatial feature engineering
* Logistic Regression-based road blockage risk estimation
* Event-level data splitting to prevent ML data leakage
* Dynamic risk-aware road graph transformation
* Configurable road severance threshold
* Capacity- and distance-aware shelter selection using CSP
* Custom Min-Heap based A* pathfinding
* Haversine geographic distance calculation
* Interactive Streamlit + Folium evacuation map
* Automated testing with Pytest

---

## System Architecture

```text
                  ┌─────────────────────┐
                  │   USGS Earthquake   │
                  │        API          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Seismic Data        │
                  │ Ingestion           │
                  └──────────┬──────────┘
                             │
                             ▼
┌─────────────────┐   ┌─────────────────────┐
│ OpenStreetMap   │──▶│ Feature Engineering │
│ + OSMnx         │   │ + Logistic Regression│
└─────────────────┘   └──────────┬──────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Road Risk P(risk)│
                       └────────┬─────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Dynamic Road Graph  │
                     │ Block / Penalize    │
                     │ Risky Roads         │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ CSP Shelter        │
                     │ Feasibility        │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Custom A* Search   │
                     │ Risk-Aware Routing │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Evacuation Route + │
                     │ Shelter Recommendation│
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Streamlit + Folium │
                     │ Interactive GUI     │
                     └─────────────────────┘
```

---

## Five-Stage Pipeline

### 1. Seismic Ingestion

Earthquake information is obtained from the **USGS API**.

Main parameters:

* Magnitude
* Epicenter latitude
* Epicenter longitude
* Depth
* Unique event ID

The project uses **1,612 regional earthquake records** for model development.

---

### 2. ML-Based Road Risk Inference

The system uses **4,839 Dhaka road segments** obtained from OpenStreetMap through OSMnx.

Earthquake and road information are combined to create approximately **1.11 million earthquake-road records**.

The Logistic Regression model uses seismic, geographic, and road-network features including:

* Earthquake magnitude
* Earthquake depth
* Epicenter-to-road distance
* Lane count
* Road classification
* Physical distance attenuation

The trained model estimates:

```text
P(risk | earthquake, road features)
```

The serialized model is stored at:

```text
models/road_risk_logistic_model.joblib
```

---

### 3. Dynamic Risk-Aware Graph

Predicted road risk is transferred to the OSMnx road network.

For roads exceeding the severance threshold:

```text
P(risk) ≥ 0.75

→ blocked = True
→ evacuation_cost = ∞
```

For traversable roads:

```text
evacuation_cost =
    length × (1 + α × P(risk)²)
```

This allows the planner to prefer a slightly longer route when it has substantially lower predicted risk.

---

### 4. Constraint Satisfaction

Candidate shelters are evaluated using a Constraint Satisfaction Problem.

Current designated shelters include:

* Dhaka Army Stadium
* Mirpur Stadium
* BUET Field
* Banani Park
* Uttara Field
* Dhanmondi Field

A shelter must satisfy:

```text
Status = OPEN
AND
Available Capacity ≥ Evacuee Group Size
AND
Haversine Distance ≤ Maximum Distance
AND
Traversable Route Exists
```

---

### 5. A* Evacuation Routing

A custom A* implementation searches the dynamically modified road graph.

The algorithm uses a **Min-Heap priority queue** and evaluates:

```text
f(n) = g(n) + h(n)
```

where:

* `g(n)` = accumulated evacuation cost
* `h(n)` = estimated remaining cost
* `f(n)` = total estimated cost

The heuristic is based on spherical Haversine distance.

---

## Mathematical Formulations

### Haversine Distance

For two geographic coordinates:

```text
a = sin²(Δφ / 2)
    + cos(φ₁) × cos(φ₂) × sin²(Δλ / 2)

c = 2 × atan2(√a, √(1-a))

d = R × c
```

where `d` is the great-circle distance.

---

### Logistic Regression

The model estimates risk using:

```text
P(y=1|x) = 1 / (1 + e⁻ᶻ)
```

where:

```text
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The model uses **L2 regularization** and class-weight balancing.

---

### CSP

The shelter-selection problem is represented as:

```text
CSP = <X, D, C>
```

where:

* `X` = shelter-selection variables
* `D` = candidate shelter domains
* `C` = capacity, status, distance, and route constraints

---

### A* Cost Function

```text
f(n) = g(n) + h(n)
```

The routing graph uses risk-adjusted edge costs so that both distance and predicted road risk influence route selection.

---

## Machine Learning Evaluation

To prevent data leakage, the dataset is split by **earthquake event ID**, rather than randomly splitting individual road records.

Reported evaluation:

| Metric                  |     Result |
| ----------------------- | ---------: |
| Accuracy                | **96.93%** |
| ROC-AUC                 | **0.9946** |
| Recall on Blocked Links |    **83%** |

The reported experiment used:

```text
Training Events: 184
Testing Events: 47
```

### Learned Feature Weights

| Feature        | Weight |
| -------------- | -----: |
| Magnitude      | +10.80 |
| Distance       |  -5.22 |
| Lane Count     |  -2.52 |
| Tertiary Roads |  +8.75 |

The coefficients provide an interpretable indication of how the corresponding features affect the model's predicted risk.

---

## Project Structure

```text
AI-Disaster-Evacuation-Planner/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── earthquakes.csv
│   │   ├── dhaka_network.graphml
│   │   ├── dhaka_edges.geojson
│   │   └── dhaka_nodes.geojson
│   │
│   └── processed/
│       ├── earthquake_road_dataset.csv
│       └── dhaka_shelters.csv
│
├── models/
│   └── road_risk_logistic_model.joblib
│
├── src/
│   ├── data/
│   │   ├── fetch_earthquakes.py
│   │   ├── fetch_roads.py
│   │   ├── create_dataset.py
│   │   └── create_shelters.py
│   │
│   ├── ml/
│   │   ├── train_model.py
│   │   └── predict_risk.py
│   │
│   ├── graph/
│   │   ├── build_graph.py
│   │   └── astar.py
│   │
│   └── csp/
│       └── shelter_csp.py
│
├── tests/
│   └── test_planner.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Technology           | Purpose                               |
| -------------------- | ------------------------------------- |
| **Python 3.10+**     | Core development                      |
| **OSMnx**            | OpenStreetMap road-network processing |
| **NetworkX**         | Graph representation                  |
| **GeoPandas**        | Geospatial processing                 |
| **Scikit-Learn**     | Machine Learning                      |
| **Streamlit**        | Interactive web application           |
| **Folium**           | Interactive maps                      |
| **Streamlit-Folium** | Folium integration                    |
| **Pytest**           | Automated testing                     |
| **Joblib**           | Model serialization                   |

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Disaster-Evacuation-Planner
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running the Application

## Streamlit Web GUI

```bash
streamlit run app/app.py
```

The dashboard provides:

* earthquake information;
* interactive Dhaka road map;
* predicted road-risk visualization;
* evacuation origin selection;
* shelter evaluation;
* risk-aware route generation;
* evacuation recommendations.

---

## Command-Line Interface

Run the complete pipeline with:

```bash
python main.py
```

---

# Testing

Run all tests:

```bash
pytest
```

Run with detailed output:

```bash
pytest -v
```

Run the planner test suite:

```bash
pytest tests/test_planner.py -v
```

### Test Coverage

The test suite verifies key components including:

* Haversine distance accuracy
* A* avoidance of blocked roads
* CSP shelter capacity enforcement

---

# Limitations

This project is an **academic decision-support prototype**, not a certified emergency-management system.

Key limitations include:

* Road-risk predictions depend on the quality of the training target and available earthquake data.
* The number of earthquake-road records is much larger than the number of independent earthquake events.
* Real earthquake damage depends on factors not fully represented by the current model.
* Shelter capacity and availability may change during an actual disaster.
* OpenStreetMap cannot guarantee real-time post-earthquake road conditions.
* Predicted road risk should not be interpreted as confirmed physical damage without authoritative observations.

---

# Future Work

Potential extensions include:

* Real-time road blockage and traffic feeds
* Live shelter occupancy
* Ground-motion intensity data
* Soil and geological information
* Infrastructure vulnerability modeling
* Bridge and building vulnerability analysis
* Multi-route evacuation recommendations
* Advanced ensemble ML models
* Real post-earthquake damage datasets
* Multi-agent evacuation simulation
* Integration with emergency-management systems

---

# Academic Contribution

The project demonstrates the integration of multiple AI and computational techniques into a single disaster-response pipeline:

```text
Machine Learning
      ↓
Risk Estimation
      ↓
Graph Transformation
      ↓
Constraint Satisfaction
      ↓
Heuristic Search
      ↓
Evacuation Planning
```

The key contribution is the integration of **ML-based road-risk inference with classical graph search and constraint-based shelter selection** to produce a risk-aware evacuation plan.

---

# License

This project is developed for **academic and research purposes**.

Third-party data and software remain subject to their respective licenses and terms of use, including data obtained from:

* United States Geological Survey (USGS)
* OpenStreetMap
* OSMnx and other open-source libraries

---

# Author

**AI-Based Earthquake Disaster Evacuation Planner — Dhaka**

Developed as an academic project focusing on:

**Artificial Intelligence · Machine Learning · Geospatial Computing · Graph Algorithms · Constraint Satisfaction · Disaster Management**
