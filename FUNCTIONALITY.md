## Causal Insight Engine - Functionality and Architecture

The Causal Insight Engine is designed as a modular, production-ready framework for end-to-end causal inference. It abstracts away the complexities of various causal estimation techniques, offering a streamlined pipeline from data ingestion and causal graph definition to robust effect estimation, validation, and deployment.

### Core Architecture

The framework is structured into several interconnected modules, each responsible for a specific stage of the causal inference pipeline:

1.  **`CausalGraph`**: Defines and manages the underlying causal assumptions.
2.  **`DataProcessor`**: Handles data preparation specific to causal analysis.
3.  **`CausalEstimator` / `EconMLEstimator`**: Interfaces with causal inference backends (DoWhy, EconML) to estimate effects.
4.  **`Refutation`**: Performs robustness checks on causal estimates.
5.  **`CausalPipeline`**: Orchestrates the entire workflow and manages the state of the analysis.

### Data Flow and Design Decisions

1.  **Data Ingestion and Preprocessing (`DataProcessor`)**
    *   **Input**: Raw `pandas.DataFrame` containing observational data.
    *   **Functionality**: Imputes missing numerical values (mean strategy) and scales numerical features (`StandardScaler`). It also performs one-hot encoding for specified categorical features.
    *   **Design Decisions**: Separating preprocessing ensures that data transformations are consistently applied and can be fitted once and then transformed across different stages (e.g., training and counterfactual simulation). `SimpleImputer` and `StandardScaler` are chosen for their robustness and common use in ML pipelines. Categorical encoding is kept simple with `get_dummies` but can be extended.

2.  **Causal Graph Definition (`CausalGraph`)**
    *   **Input**: A list of nodes (variables) and directed edges (causal relationships).
    *   **Functionality**: Builds a directed acyclic graph (DAG) using `networkx` to represent the hypothesized causal structure. This graph is crucial for identifying the causal estimand.
    *   **Design Decisions**: `networkx` provides a robust and flexible way to represent graphs. The graph can be visualized to aid in understanding and validating assumptions. This module is foundational as it allows users to explicitly encode their domain knowledge about causal pathways, which is critical for valid inference.

3.  **Causal Effect Identification and Estimation (`CausalEstimator`, `EconMLEstimator`)**
    *   **Input**: Processed `DataFrame`, defined `treatment`, `outcome`, `common_causes`, `instruments`, `effect_modifiers`, and the causal graph (if available).
    *   **Functionality**: 
        *   **`CausalEstimator` (DoWhy-backed)**: Utilizes the `dowhy.CausalModel` to first *identify* the causal estimand (i.e., translate the causal question into a statistical estimand based on the graph) and then *estimate* the effect using various methods (e.g., G-Formula, IV, Propensity Score Matching). It abstracts `DoWhy`'s estimation interface.
        *   **`EconMLEstimator` (EconML-backed)**: Leverages advanced causal ML estimators from `EconML` such as Double Machine Learning (DML) and Causal Forests. These methods use ML models (e.g., `MLPRegressor`) for nuisance parameter estimation, offering flexibility and robustness to model misspecification. This estimator directly uses `EconML`'s `fit` and `effect` methods.
    *   **Design Decisions**: 
        *   **Abstraction**: A common interface (`CausalEstimator`) is used, with specialized subclasses for different backend libraries (`DoWhy`, `EconML`). This allows users to switch between different causal estimation philosophies and methods without rewriting their entire pipeline.
        *   **DoWhy Integration**: `DoWhy` is excellent for its four-step approach (Model, Identify, Estimate, Refute) and its graph-based identification, which provides strong theoretical guarantees.
        *   **EconML Integration**: `EconML` provides state-of-the-art causal machine learning techniques that are highly flexible and robust, especially for heterogeneous treatment effects. Using `MLPRegressor` as default models for `model_y` and `model_t` ensures a modern, flexible baseline, but users can provide any scikit-learn compatible regressor.
        *   **Explicit Confounders/Effect Modifiers**: While `DoWhy` can infer common causes from a graph, explicitly listing them for `EconML` methods ensures clarity and control, aligning with how these libraries typically operate.

4.  **Robustness Checks (`Refutation`)**
    *   **Input**: The `CausalModel` object, identified estimand, and the estimated effect.
    *   **Functionality**: Implements various refutation tests provided by `DoWhy`. These tests challenge the validity of the causal estimate by modifying the data or model assumptions (e.g., adding a random confounder, using a placebo treatment, re-estimating on subsets).
    *   **Design Decisions**: Refutation is critical for building trust in causal estimates. Integrating `DoWhy`'s comprehensive refutation suite ensures that the results are rigorously tested for sensitivity to unmeasured confounders or model choices. This moves the pipeline beyond just getting an estimate to *validating* it.

5.  **Counterfactual Simulation (`CausalPipeline` method - conceptual)**
    *   **Input**: Intervention data (a DataFrame representing a hypothetical world where treatment is set).
    *   **Functionality (conceptual)**: Predicts the outcome under a hypothetical intervention (e.g., 