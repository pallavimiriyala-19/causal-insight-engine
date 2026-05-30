import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
import networkx as nx
import dowhy.gcm as gcm
from dowhy import CausalModel
import econml.dml as dml
from econml.grf import CausalForestDML
from econml.dr import LinearDRLearner
from typing import List, Dict, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CausalGraph:
    """Manages the definition and visualization of causal graphs."""
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str]]):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        logger.info(f"Causal graph initialized with {len(nodes)} nodes and {len(edges)} edges.")

    def add_edge(self, u: str, v: str):
        if u not in self.graph.nodes or v not in self.graph.nodes:
            logger.error(f"One or both nodes {u}, {v} not in graph. Edge not added.")
            raise ValueError("Nodes must exist in the graph to add an edge.")
        self.graph.add_edge(u, v)
        logger.debug(f"Added edge from {u} to {v}.")

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def draw(self, path: Optional[str] = None):
        """Draws the causal graph using Matplotlib. Requires pygraphviz for better layout."""
        try:
            import matplotlib.pyplot as plt
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        except ImportError:
            logger.warning("PyGraphviz not found, using default networkx layout. Install 'pygraphviz' for better graph visualizations.")
            pos = nx.spring_layout(self.graph)
        except Exception as e:
            logger.warning(f"Error with graphviz_layout ({e}), falling back to spring_layout.")
            pos = nx.spring_layout(self.graph)

        plt.figure(figsize=(10, 8))
        nx.draw_networkx(self.graph, pos, with_labels=True, node_color='lightblue', 
                         edge_color='gray', arrows=True, node_size=2000, font_size=10,
                         arrowstyle='-|>', arrowsize=20)
        plt.title("Causal Graph")
        if path:
            plt.savefig(path)
            logger.info(f"Causal graph saved to {path}")
        plt.show()


class DataProcessor:
    """Handles data preprocessing for causal inference, including imputation and scaling."""
    def __init__(self, numerical_features: List[str], categorical_features: List[str] = None):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        logger.info(f"DataProcessor initialized for numerical features: {numerical_features}")

    def fit(self, df: pd.DataFrame):
        """Fits the imputer and scaler to the numerical features of the DataFrame."""
        if not self.numerical_features: # If no numerical features, do nothing
            return
        numerical_data = df[self.numerical_features]
        self.imputer.fit(numerical_data)
        imputed_data = self.imputer.transform(numerical_data)
        self.scaler.fit(imputed_data)
        logger.info("DataProcessor fit completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the DataFrame by imputing missing values and scaling numerical features."""
        df_processed = df.copy()
        
        if self.numerical_features:
            numerical_data = df_processed[self.numerical_features]
            if self.imputer: # Ensure imputer is fitted
                numerical_data_imputed = self.imputer.transform(numerical_data)
                df_processed[self.numerical_features] = numerical_data_imputed
            else:
                logger.warning("Imputer not fitted, skipping imputation for numerical features.")
            
            if self.scaler: # Ensure scaler is fitted
                scaled_data = self.scaler.transform(df_processed[self.numerical_features])
                df_processed[self.numerical_features] = scaled_data
            else:
                logger.warning("Scaler not fitted, skipping scaling for numerical features.")
        
        # Handle categorical features - simple one-hot encoding for now
        if self.categorical_features:
            df_processed = pd.get_dummies(df_processed, columns=self.categorical_features, drop_first=True)
            logger.debug(f"Categorical features {self.categorical_features} one-hot encoded.")

        logger.info("DataProcessor transform completed.")
        return df_processed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def inverse_transform_numerical(self, df_scaled: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Inverse transforms scaled numerical features to their original scale."""
        df_original = df_scaled.copy()
        if not self.scaler:
            logger.warning("Scaler not fitted, cannot inverse transform.")
            return df_original
        
        if not all(f in df_scaled.columns for f in features):
            logger.error(f"Not all specified features {features} are in the scaled DataFrame.")
            raise ValueError("Features for inverse transformation not found.")

        df_original[features] = self.scaler.inverse_transform(df_scaled[features])
        logger.info("Inverse transformation of numerical features completed.")
        return df_original


class CausalEstimator:
    """Abstract base class for causal estimators."""
    def __init__(self, model_identifier: str, treatment: str, outcome: str, 
                 common_causes: Optional[List[str]] = None, 
                 instruments: Optional[List[str]] = None, 
                 effect_modifiers: Optional[List[str]] = None,
                 causal_graph_dot: Optional[str] = None):
        self.model_identifier = model_identifier
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes if common_causes is not None else []
        self.instruments = instruments if instruments is not None else []
        self.effect_modifiers = effect_modifiers if effect_modifiers is not None else []
        self.causal_graph_dot = causal_graph_dot
        self.causal_model: Optional[CausalModel] = None
        self.identified_estimand: Optional[Any] = None
        self.estimate: Optional[Any] = None
        logger.info(f"CausalEstimator '{model_identifier}' initialized for T:{treatment}, Y:{outcome}.")

    def _build_dowhy_model(self, df: pd.DataFrame):
        """Builds the DoWhy CausalModel object."""
        if self.causal_graph_dot:
            self.causal_model = CausalModel(data=df,
                                           treatment=self.treatment,
                                           outcome=self.outcome,
                                           graph=self.causal_graph_dot)
        else:
            # If no graph, DoWhy attempts to infer common causes. 
            # Explicitly pass common_causes for better control.
            self.causal_model = CausalModel(data=df,
                                           treatment=self.treatment,
                                           outcome=self.outcome,
                                           common_causes=self.common_causes,
                                           instruments=self.instruments,
                                           effect_modifiers=self.effect_modifiers)
        logger.debug("DoWhy CausalModel built.")

    def identify_estimand(self):
        """Identifies the causal estimand using DoWhy's graph-based criteria."""
        if self.causal_model is None:
            raise ValueError("CausalModel not built. Call _build_dowhy_model first.")
        self.identified_estimand = self.causal_model.identify_effect(proceed_when_unidentifiable=True)
        logger.info(f"Causal estimand identified: {self.identified_estimand.estimand_expression}")
        return self.identified_estimand

    def estimate_effect(self, df: pd.DataFrame, method_name: str, control_value: Any = 0, treatment_value: Any = 1, 
                        target_units: str = "ate", 
                        confidence_intervals: bool = True,
                        test_significance: bool = True,
                        **kwargs):
        """Estimates the causal effect using the specified method."""
        if self.causal_model is None or self.identified_estimand is None:
            self._build_dowhy_model(df)
            self.identify_estimand()
            
        logger.info(f"Estimating causal effect using {method_name}...")
        self.estimate = self.causal_model.estimate_effect(self.identified_estimand,
                                                          method_name=method_name,
                                                          control_value=control_value,
                                                          treatment_value=treatment_value,
                                                          target_units=target_units,
                                                          confidence_intervals=confidence_intervals,
                                                          test_significance=test_significance,
                                                          **kwargs)
        logger.info(f"Causal effect estimated. Value: {self.estimate.value}")
        return self.estimate
    
    def get_estimate(self) -> Optional[Any]:
        return self.estimate


class EconMLEstimator(CausalEstimator):
    """A CausalEstimator specifically for EconML-backed methods."""
    def __init__(self, model_identifier: str, treatment: str, outcome: str, 
                 common_causes: Optional[List[str]] = None, 
                 instruments: Optional[List[str]] = None, 
                 effect_modifiers: Optional[List[str]] = None,
                 causal_graph_dot: Optional[str] = None,
                 estimator_type: str = "dml", # dml, drlearner, causal_forest
                 model_y: Any = None, model_t: Any = None, model_final: Any = None,
                 **estimator_kwargs):
        super().__init__(model_identifier, treatment, outcome, common_causes, 
                         instruments, effect_modifiers, causal_graph_dot)
        self.estimator_type = estimator_type
        self.model_y = model_y if model_y is not None else MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200)
        self.model_t = model_t if model_t is not None else MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200)
        self.model_final = model_final if model_final is not None else MLPRegressor(hidden_layer_sizes=(50,), max_iter=100)
        self.estimator_kwargs = estimator_kwargs
        self.econml_learner = None
        logger.info(f"EconMLEstimator '{model_identifier}' configured with type: {estimator_type}.")

    def _initialize_econml_learner(self, W: pd.DataFrame, X: pd.DataFrame):
        if self.estimator_type == "dml":
            self.econml_learner = dml.LinearDML(model_y=self.model_y,
                                                model_t=self.model_t,
                                                **self.estimator_kwargs)
        elif self.estimator_type == "drlearner":
            self.econml_learner = LinearDRLearner(model_regression=self.model_y,
                                                  model_propensity=self.model_t,
                                                  model_final=self.model_final,
                                                  **self.estimator_kwargs)
        elif self.estimator_type == "causal_forest":
            self.econml_learner = CausalForestDML(model_y=self.model_y,
                                                  model_t=self.model_t,
                                                  **self.estimator_kwargs)
        else:
            raise ValueError(f"Unknown estimator_type: {self.estimator_type}")
        logger.debug(f"EconML learner initialized: {type(self.econml_learner).__name__}")

    def estimate_effect(self, df: pd.DataFrame, 
                        X_features: Optional[List[str]] = None, 
                        W_features: Optional[List[str]] = None,
                        **kwargs) -> Any:
        """Estimates the causal effect using EconML methods."""
        if X_features is None: X_features = [] # Effect modifiers
        if W_features is None: W_features = self.common_causes # Common causes/confounders
        
        # Ensure all required columns are present
        required_cols = [self.treatment, self.outcome] + X_features + W_features
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Missing required columns for EconML estimation: {missing}")
            raise ValueError(f"DataFrame is missing columns: {missing}")

        T = df[self.treatment]
        Y = df[self.outcome]
        X = df[X_features] if X_features else None # Effect modifiers
        W = df[W_features] if W_features else None # Confounders
        
        self._initialize_econml_learner(W, X)
        
        logger.info(f"Fitting EconML {self.estimator_type} model...")
        self.econml_learner.fit(Y, T, X=X, W=W)
        logger.info(f"EconML {self.estimator_type} model fitted.")

        # For overall ATE (Average Treatment Effect)
        # If X is present, we predict for a 'typical' individual or average across them
        if X is not None and not X.empty:
            # Predict for a specific X or average over X to get an overall ATE
            # For simplicity, let's predict at the mean of X for ATE if X is present
            X_test = pd.DataFrame([X.mean()], columns=X.columns) if X is not None else None
        else:
            X_test = None

        self.estimate = self.econml_learner.const_marginal_effect(X_test) if X_test is not None else self.econml_learner.const_marginal_effect()
        
        logger.info(f"Causal effect estimated (EconML). Value: {np.mean(self.estimate)}")
        return np.mean(self.estimate) # Return average ATE

    def predict_individual_effect(self, df_new: pd.DataFrame, X_features: Optional[List[str]] = None, W_features: Optional[List[str]] = None) -> np.ndarray:
        """Predicts Conditional Average Treatment Effect (CATE) for new data."""
        if self.econml_learner is None:
            raise ValueError("EconML learner not fitted. Call estimate_effect first.")
        
        if X_features is None: X_features = []
        if W_features is None: W_features = self.common_causes

        X_new = df_new[X_features] if X_features else None
        W_new = df_new[W_features] if W_features else None # W might be needed for prediction for some methods

        logger.info(f"Predicting individual causal effects using EconML {self.estimator_type} model...")
        cate = self.econml_learner.effect(X_new) # Predict CATE for each row in df_new
        return cate


class Refutation:
    """Performs refutation tests on causal estimates to check robustness."""
    def __init__(self, causal_model: CausalModel, estimand: Any, estimate: Any):
        self.causal_model = causal_model
        self.estimand = estimand
        self.estimate = estimate
        logger.info("Refutation module initialized.")

    def refute_random_common_cause(self, num_simulations: int = 10, **kwargs) -> Any:
        """Adds a randomly generated confounder to check robustness."""
        logger.info(f"Performing refutation: Adding a random common cause ({num_simulations} simulations)...")
        refute_result = self.causal_model.refute_estimate(self.estimand, self.estimate, 
                                                           method_name="random_common_cause",
                                                           num_simulations=num_simulations,
                                                           **kwargs)
        logger.info(f"Random common cause refutation result: {refute_result.refutation_result}")
        return refute_result

    def refute_placebo_treatment(self, placebo_type: str = "randomize_treatment", num_simulations: int = 10, **kwargs) -> Any:
        """Replaces the treatment with a placebo and checks if the effect disappears."""
        logger.info(f"Performing refutation: Placebo treatment ({placebo_type}, {num_simulations} simulations)...")
        refute_result = self.causal_model.refute_estimate(self.estimand, self.estimate, 
                                                           method_name="placebo_treatment",
                                                           placebo_type=placebo_type,
                                                           num_simulations=num_simulations,
                                                           **kwargs)
        logger.info(f"Placebo treatment refutation result: {refute_result.refutation_result}")
        return refute_result

    def refute_data_subset_permutations(self, subset_fraction: float = 0.8, num_permutations: int = 10, **kwargs) -> Any:
        """Checks robustness by re-estimating on subsets of data."""
        logger.info(f"Performing refutation: Data subset permutations (fraction={subset_fraction}, {num_permutations} permutations)...")
        refute_result = self.causal_model.refute_estimate(self.estimand, self.estimate, 
                                                           method_name="data_subset_refuter",
                                                           subset_fraction=subset_fraction,
                                                           num_permutations=num_permutations,
                                                           **kwargs)
        logger.info(f"Data subset refutation result: {refute_result.refutation_result}")
        return refute_result


class CausalPipeline:
    """Orchestrates the entire causal inference pipeline from data to deployment."""
    def __init__(self, name: str):
        self.name = name
        self.data_processor: Optional[DataProcessor] = None
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_estimator: Optional[Union[CausalEstimator, EconMLEstimator]] = None
        self.fitted_df: Optional[pd.DataFrame] = None
        self.estimand: Optional[Any] = None
        self.estimate: Optional[Any] = None
        logger.info(f"CausalPipeline '{name}' initialized.")

    def setup_data_processor(self, numerical_features: List[str], categorical_features: List[str] = None):
        self.data_processor = DataProcessor(numerical_features, categorical_features)
        logger.info("Data processor configured.")

    def setup_causal_graph(self, nodes: List[str], edges: List[Tuple[str, str]]):
        self.causal_graph = CausalGraph(nodes, edges)
        logger.info("Causal graph configured.")

    def setup_estimator(self, treatment: str, outcome: str, 
                        common_causes: Optional[List[str]] = None, 
                        instruments: Optional[List[str]] = None, 
                        effect_modifiers: Optional[List[str]] = None,
                        estimator_type: str = "dowhy_gformula", # dowhy_gformula, dowhy_iv, econml_dml, econml_causal_forest
                        model_y: Any = None, model_t: Any = None, model_final: Any = None,
                        **estimator_kwargs):
        
        graph_dot = None
        if self.causal_graph:
            graph_dot = nx.nx_pydot.to_pydot(self.causal_graph.get_graph()).to_string()

        if estimator_type.startswith("dowhy"):
            self.causal_estimator = CausalEstimator(model_identifier=estimator_type, treatment=treatment, 
                                                    outcome=outcome, common_causes=common_causes, 
                                                    instruments=instruments, effect_modifiers=effect_modifiers,
                                                    causal_graph_dot=graph_dot)
        elif estimator_type.startswith("econml"):
            econml_type = estimator_type.replace("econml_", "")
            self.causal_estimator = EconMLEstimator(model_identifier=estimator_type, treatment=treatment, 
                                                    outcome=outcome, common_causes=common_causes, 
                                                    instruments=instruments, effect_modifiers=effect_modifiers,
                                                    causal_graph_dot=graph_dot, estimator_type=econml_type,
                                                    model_y=model_y, model_t=model_t, model_final=model_final,
                                                    **estimator_kwargs)
        else:
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")
        logger.info(f"Causal estimator configured as {estimator_type}.")

    def run_pipeline(self, df: pd.DataFrame, 
                     dowhy_method_name: Optional[str] = None, 
                     econml_X_features: Optional[List[str]] = None, 
                     econml_W_features: Optional[List[str]] = None,
                     **estimator_kwargs):
        """Executes the causal inference pipeline steps."""
        if self.data_processor is None or self.causal_estimator is None:
            raise ValueError("Data processor and causal estimator must be set up first.")

        logger.info("Starting data preprocessing...")
        self.fitted_df = self.data_processor.fit_transform(df)
        logger.info("Data preprocessing complete.")

        if isinstance(self.causal_estimator, CausalEstimator) and not isinstance(self.causal_estimator, EconMLEstimator):
            # DoWhy based estimator
            self.causal_estimator._build_dowhy_model(self.fitted_df)
            self.estimand = self.causal_estimator.identify_estimand()
            self.estimate = self.causal_estimator.estimate_effect(self.fitted_df, method_name=dowhy_method_name, **estimator_kwargs)
        elif isinstance(self.causal_estimator, EconMLEstimator):
            # EconML based estimator
            # Note: EconML estimators might not strictly use the DoWhy graph for identification, but rather the passed W/X
            # For consistency, we still build the DoWhy model and identify if graph is provided.
            if self.causal_graph:
                graph_dot = nx.nx_pydot.to_pydot(self.causal_graph.get_graph()).to_string()
                dowhy_model = CausalModel(data=self.fitted_df, 
                                          treatment=self.causal_estimator.treatment,
                                          outcome=self.causal_estimator.outcome,
                                          graph=graph_dot)
                self.estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
            else:
                logger.warning("No causal graph provided for EconML estimator. Estimand identification skipped based on graph.")
                # Create a dummy estimand for consistency if required later, or handle it differently
                self.estimand = None

            self.estimate = self.causal_estimator.estimate_effect(self.fitted_df, X_features=econml_X_features, W_features=econml_W_features, **estimator_kwargs)
            
        logger.info("Causal pipeline execution complete.")
        return self.estimate

    def get_causal_effect(self) -> Optional[Any]:
        """Returns the estimated causal effect."""
        if self.estimate is None:
            logger.warning("No causal effect has been estimated yet.")
        return self.estimate

    def perform_refutation(self, refutation_type: str = "random_common_cause", 
                           num_simulations: int = 10, **kwargs) -> Any:
        """Performs a refutation test on the current causal estimate."""
        if self.causal_estimator is None or self.estimand is None or self.estimate is None:
            raise ValueError("Estimator, estimand, or estimate are not available for refutation. Run the pipeline first.")
        
        # Ensure the DoWhy CausalModel is available for refutation. Rebuild if using EconML without initial DoWhy model.
        dowhy_model_for_refutation = None
        if isinstance(self.causal_estimator, CausalEstimator) and self.causal_estimator.causal_model:
            dowhy_model_for_refutation = self.causal_estimator.causal_model
        elif self.causal_graph and self.fitted_df is not None: # Rebuild DoWhy model for refutation if EconML was used
             graph_dot = nx.nx_pydot.to_pydot(self.causal_graph.get_graph()).to_string()
             dowhy_model_for_refutation = CausalModel(data=self.fitted_df, 
                                                      treatment=self.causal_estimator.treatment,
                                                      outcome=self.causal_estimator.outcome,
                                                      graph=graph_dot)
        else:
             raise ValueError("Cannot perform refutation without a DoWhy CausalModel or a graph definition.")

        refuter = Refutation(dowhy_model_for_refutation, self.estimand, self.estimate)

        if refutation_type == "random_common_cause":
            return refuter.refute_random_common_cause(num_simulations=num_simulations, **kwargs)
        elif refutation_type == "placebo_treatment":
            return refuter.refute_placebo_treatment(num_simulations=num_simulations, **kwargs)
        elif refutation_type == "data_subset_permutations":
            return refuter.refute_data_subset_permutations(num_permutations=num_simulations, **kwargs)
        else:
            raise ValueError(f"Unsupported refutation type: {refutation_type}")

    def simulate_counterfactual(self, intervention_data: pd.DataFrame) -> pd.DataFrame:
        """Simulates counterfactual outcomes based on interventions."""
        if self.causal_estimator is None or self.fitted_df is None:
            raise ValueError("Estimator or fitted data not available. Run the pipeline first.")

        if not isinstance(self.causal_estimator, CausalEstimator) or not self.causal_estimator.causal_model:
            raise ValueError("Counterfactual simulation currently requires a DoWhy-based CausalModel.")

        logger.info("Simulating counterfactuals...")
        # DoWhy's built-in GCM for counterfactuals is powerful.
        # This is a simplification; a full GCM approach would involve more setup.
        # For this example, we'll assume the CausalModel itself can be used to predict if an intervention is defined.
        # A more robust solution might involve: gcm.StructuralCausalModel, gcm.fit, gcm.counterfactual_manipulation

        # For direct counterfactuals, we can use the fitted estimator's prediction capabilities.
        # This highly depends on the specific estimator and its support for counterfactuals.
        # A general approach is difficult without making strong assumptions.
        # Let's outline a conceptual approach for a simple case or direct DoWhy GCM integration.

        # Option 1: Using DoWhy's GCM for counterfactuals (more robust)
        # Requires fitting a GCM first. This is a separate module in DoWhy.
        # For now, let's keep it conceptual or use a simplified approach.
        
        # Placeholder: If the estimator provides a direct way to compute Y(do(T=t))
        # For a basic example, if we want to see Y if T was always 'treatment_value'
        # This requires more specific implementation based on the estimator type.
        logger.warning("Direct counterfactual simulation for general estimators is complex. Implement specific logic based on estimator_type.")
        logger.info("Returning intervention data with original outcome for now. Advanced counterfactuals require specific model support.")
        return intervention_data # placeholder

    def save_pipeline(self, path: str):
        """Saves the entire pipeline (processor, graph, estimator) to disk."""
        pipeline_state = {
            'name': self.name,
            'data_processor': self.data_processor,
            'causal_graph': self.causal_graph,
            'causal_estimator': self.causal_estimator,
            'estimand': self.estimand,
            'estimate': self.estimate
        }
        joblib.dump(pipeline_state, path)
        logger.info(f"Causal pipeline '{self.name}' saved to {path}")

    @staticmethod
    def load_pipeline(path: str) -> 'CausalPipeline':
        """Loads a saved pipeline from disk."""
        pipeline_state = joblib.load(path)
        loaded_pipeline = CausalPipeline(pipeline_state['name'])
        loaded_pipeline.data_processor = pipeline_state['data_processor']
        loaded_pipeline.causal_graph = pipeline_state['causal_graph']
        loaded_pipeline.causal_estimator = pipeline_state['causal_estimator']
        loaded_pipeline.estimand = pipeline_state['estimand']
        loaded_pipeline.estimate = pipeline_state['estimate']
        logger.info(f"Causal pipeline '{loaded_pipeline.name}' loaded from {path}")
        return loaded_pipeline

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the pipeline state and results."""
        summary = {
            "Pipeline Name": self.name,
            "Data Processor Configured": self.data_processor is not None,
            "Causal Graph Configured": self.causal_graph is not None,
            "Causal Estimator Configured": self.causal_estimator.model_identifier if self.causal_estimator else "None",
            "Treatment Variable": self.causal_estimator.treatment if self.causal_estimator else "N/A",
            "Outcome Variable": self.causal_estimator.outcome if self.causal_estimator else "N/A",
            "Identified Estimand": str(self.estimand.estimand_expression) if self.estimand else "N/A",
            "Estimated Causal Effect": self.estimate.value if self.estimate and hasattr(self.estimate, 'value') else (
                np.mean(self.estimate) if self.estimate is not None and isinstance(self.estimate, np.ndarray) else "N/A"
            ),
            "Effect Confidence Intervals": self.estimate.get_confidence_intervals() if self.estimate and hasattr(self.estimate, 'get_confidence_intervals') else "N/A",
            "Effect Significance": self.estimate.get_p_value() if self.estimate and hasattr(self.estimate, 'get_p_value') else "N/A"
        }
        return summary


# Example usage outside of the class for demonstration purposes
if __name__ == '__main__':
    logger.info("Running Causal Insight Engine example...")

    # 1. Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    education = np.random.randint(0, 3, n_samples) # 0: high school, 1: college, 2: grad
    
    # Treatment: Marketing Campaign (binary)
    # Likelihood of treatment depends on age and income (confounders)
    treatment_prob = 1 / (1 + np.exp(-(0.05 * age - 0.00002 * income + np.random.normal(0, 0.5, n_samples))))
    marketing_campaign = (np.random.rand(n_samples) < treatment_prob).astype(int)

    # Outcome: Purchase Amount
    # Purchase amount depends on marketing campaign, age, income, and education
    purchase_amount = (50 + 20 * marketing_campaign + 0.8 * age + 0.0001 * income + 
                       10 * education + np.random.normal(0, 20, n_samples))

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'marketing_campaign': marketing_campaign,
        'purchase_amount': purchase_amount
    })
    df['education'] = df['education'].astype('category') # Treat education as categorical

    logger.info("Synthetic data generated.")
    logger.info(df.head())

    # 2. Initialize the Causal Pipeline
    pipeline = CausalPipeline(name="Marketing_Impact_Analysis")

    # 3. Setup Data Processor
    numerical_features = ['age', 'income', 'purchase_amount']
    categorical_features = ['education']
    pipeline.setup_data_processor(numerical_features, categorical_features)

    # 4. Setup Causal Graph
    # Define the causal relationships explicitly
    # age -> marketing_campaign, age -> purchase_amount
    # income -> marketing_campaign, income -> purchase_amount
    # education -> purchase_amount
    # marketing_campaign -> purchase_amount
    nodes = ['age', 'income', 'education', 'marketing_campaign', 'purchase_amount']
    edges = [
        ('age', 'marketing_campaign'), ('age', 'purchase_amount'),
        ('income', 'marketing_campaign'), ('income', 'purchase_amount'),
        ('education', 'purchase_amount'),
        ('marketing_campaign', 'purchase_amount')
    ]
    pipeline.setup_causal_graph(nodes, edges)
    pipeline.causal_graph.draw(path='causal_graph.png')

    # 5. Setup Causal Estimator (using DoWhy's G-Formula for Average Treatment Effect)
    treatment = 'marketing_campaign'
    outcome = 'purchase_amount'
    # Common causes are inferred from the graph or explicitly passed if no graph
    common_causes = ['age', 'income', 'education'] # For clarity, explicitly listing confounders
    
    # Example 1: Using a DoWhy-backed estimator (e.g., G-Formula)
    pipeline.setup_estimator(treatment=treatment, outcome=outcome,
                             common_causes=common_causes,
                             estimator_type="dowhy_gformula")
    logger.info("Running pipeline with DoWhy G-Formula...")
    dowhy_estimate = pipeline.run_pipeline(df.copy(), dowhy_method_name="gformula")
    print(f"\nDoWhy Estimated Causal Effect (G-Formula): {dowhy_estimate.value:.2f}")
    print(f"Confidence Interval: {dowhy_estimate.get_confidence_intervals()}")
    print(f"P-value: {dowhy_estimate.get_p_value()}")

    # Perform refutation
    logger.info("\nPerforming refutation: Random Common Cause...")
    refute_result = pipeline.perform_refutation(refutation_type="random_common_cause", num_simulations=5)
    print(f"Refutation (Random Common Cause): {refute_result.refutation_result}")

    # Example 2: Using an EconML-backed estimator (e.g., Linear DML)
    # Reset pipeline for new estimator
    pipeline_econml = CausalPipeline(name="Marketing_Impact_Analysis_EconML")
    pipeline_econml.setup_data_processor(numerical_features, categorical_features)
    pipeline_econml.setup_causal_graph(nodes, edges)

    # Define simple regressors for EconML. You could use more complex models like RandomForestRegressor.
    model_y_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    model_t_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)

    pipeline_econml.setup_estimator(treatment=treatment, outcome=outcome,
                                    common_causes=common_causes,
                                    estimator_type="econml_dml",
                                    model_y=model_y_reg, model_t=model_t_reg,
                                    cv=2 # K-fold cross-validation for DML
                                    )
    logger.info("Running pipeline with EconML Linear DML...")
    # Note: For EconML, W_features are confounders, X_features are effect modifiers. 
    # Here we treat education as a confounder, but it could be an effect modifier too.
    econml_estimate = pipeline_econml.run_pipeline(df.copy(), 
                                                   econml_W_features=['age', 'income', 'education'] # Confounders
                                                   )
    print(f"\nEconML Estimated Causal Effect (Linear DML): {econml_estimate:.2f}")
    
    # Predict CATE (Conditional Average Treatment Effect) for a subset of data
    sample_df = df.sample(5, random_state=42)
    cate_predictions = pipeline_econml.causal_estimator.predict_individual_effect(sample_df, 
                                                                               X_features=[], # No effect modifiers for this example
                                                                               W_features=['age', 'income', 'education'])
    print("\nCATE predictions for sample data:")
    print(sample_df[['age', 'income', 'education', 'marketing_campaign', 'purchase_amount']].to_string())
    print(f"Predicted CATEs: {cate_predictions.flatten()}")

    # 6. Save and Load Pipeline
    pipeline_path = "marketing_causal_pipeline.joblib"
    pipeline_econml.save_pipeline(pipeline_path)
    loaded_pipeline = CausalPipeline.load_pipeline(pipeline_path)
    print(f"\nLoaded pipeline summary: {loaded_pipeline.get_summary()}")

    logger.info("Causal Insight Engine example finished.")