import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import joblib
import networkx as nx
from dowhy import CausalModel
from econml.dml import LinearDML

# Assuming causal_insight_engine.py is in the same directory or installed
from causal_insight_engine import CausalPipeline, CausalGraph, DataProcessor, CausalEstimator, EconMLEstimator, Refutation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Running Causal Insight Engine full example demo...")

    # 1. Generate synthetic data for a marketing campaign impact scenario
    np.random.seed(42)
    n_samples = 2000
    age = np.random.normal(35, 8, n_samples)
    income = np.random.normal(60000, 20000, n_samples)
    prior_purchases = np.random.randint(0, 5, n_samples)
    
    # Treatment: Marketing Campaign (binary)
    # Likelihood of treatment depends on age, income, and prior_purchases (confounders)
    treatment_prob_logit = -2 + 0.07 * age - 0.00001 * income + 0.5 * prior_purchases + np.random.normal(0, 0.8, n_samples)
    treatment_prob = 1 / (1 + np.exp(-treatment_prob_logit))
    marketing_campaign = (np.random.rand(n_samples) < treatment_prob).astype(int)

    # Outcome: Customer Lifetime Value (CLV)
    # CLV depends on marketing campaign, age, income, prior_purchases
    customer_lifetime_value = (
        100 + 50 * marketing_campaign + 1.2 * age + 0.0002 * income + 
        30 * prior_purchases + np.random.normal(0, 50, n_samples)
    )

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'prior_purchases': prior_purchases,
        'marketing_campaign': marketing_campaign,
        'customer_lifetime_value': customer_lifetime_value
    })

    logger.info("Synthetic data generated.")
    logger.info(df.head())
    logger.info(f"Dataset shape: {df.shape}")

    # 2. Initialize the Causal Pipeline
    pipeline_name = "Customer_CLV_Causal_Analysis"
    pipeline = CausalPipeline(name=pipeline_name)

    # 3. Setup Data Processor
    numerical_features = ['age', 'income', 'prior_purchases', 'customer_lifetime_value']
    categorical_features = [] # No categorical features in this synthetic example
    pipeline.setup_data_processor(numerical_features, categorical_features)

    # 4. Setup Causal Graph
    # Define the causal relationships explicitly
    # age -> marketing_campaign, age -> customer_lifetime_value
    # income -> marketing_campaign, income -> customer_lifetime_value
    # prior_purchases -> marketing_campaign, prior_purchases -> customer_lifetime_value
    # marketing_campaign -> customer_lifetime_value
    nodes = ['age', 'income', 'prior_purchases', 'marketing_campaign', 'customer_lifetime_value']
    edges = [
        ('age', 'marketing_campaign'), ('age', 'customer_lifetime_value'),
        ('income', 'marketing_campaign'), ('income', 'customer_lifetime_value'),
        ('prior_purchases', 'marketing_campaign'), ('prior_purchases', 'customer_lifetime_value'),
        ('marketing_campaign', 'customer_lifetime_value')
    ]
    pipeline.setup_causal_graph(nodes, edges)
    pipeline.causal_graph.draw(path='clv_causal_graph.png')
    logger.info("Causal graph defined and visualized.")

    # 5. Setup Causal Estimator: We'll use EconML's LinearDML for its robustness to confounders
    treatment = 'marketing_campaign'
    outcome = 'customer_lifetime_value'
    # Common causes (confounders) that influence both treatment and outcome
    common_causes = ['age', 'income', 'prior_purchases'] 
    effect_modifiers = [] # Variables that modify the treatment effect (e.g., segmenting by age)

    # Define base ML models for EconML's DML estimator
    # Using simple MLP regressors for demonstration. For production, consider RandomForest, GradientBoosting, etc.
    model_y_base = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42, early_stopping=True)
    model_t_base = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42, early_stopping=True)

    pipeline.setup_estimator(treatment=treatment, outcome=outcome,
                             common_causes=common_causes,
                             effect_modifiers=effect_modifiers,
                             estimator_type="econml_dml", # Using EconML DML
                             model_y=model_y_base, model_t=model_t_base,
                             cv=3 # 3-fold cross-validation for DML, important for robustness
                            )
    logger.info("Causal estimator (EconML Linear DML) configured.")

    # 6. Run the Causal Pipeline
    logger.info("Executing the causal pipeline...")
    # For EconML, specify confounders (W_features) and effect modifiers (X_features)
    estimated_effect = pipeline.run_pipeline(df.copy(), 
                                            econml_W_features=common_causes,
                                            econml_X_features=effect_modifiers
                                            )
    
    print(f"\nEstimated Average Causal Effect (EconML Linear DML) of 'marketing_campaign' on 'customer_lifetime_value': {estimated_effect:.2f}")

    # 7. Perform Refutation Tests
    # Note: Refutation methods currently use DoWhy's CausalModel internally. 
    # The CausalPipeline class handles rebuilding this if an EconML estimator was used initially.
    logger.info("\nPerforming refutation tests...")
    
    # Refutation 1: Add a random common cause (unobserved confounder)
    refute_random_common_cause_result = pipeline.perform_refutation("random_common_cause", num_simulations=5)
    print(f"Refutation (Random Common Cause): {refute_random_common_cause_result.refutation_result}")
    print(f"  Original estimate: {refute_random_common_cause_result.initial_effect:.2f}")
    print(f"  New estimate: {refute_random_common_cause_result.new_effect.mean():.2f}")

    # Refutation 2: Use a placebo treatment
    refute_placebo_result = pipeline.perform_refutation("placebo_treatment", placebo_type="randomize_treatment", num_simulations=5)
    print(f"Refutation (Placebo Treatment): {refute_placebo_result.refutation_result}")
    print(f"  Original estimate: {refute_placebo_result.initial_effect:.2f}")
    print(f"  New estimate: {refute_placebo_result.new_effect.mean():.2f}")

    # Refutation 3: Re-estimate on data subsets
    refute_subset_result = pipeline.perform_refutation("data_subset_permutations", subset_fraction=0.7, num_simulations=5)
    print(f"Refutation (Data Subset Permutations): {refute_subset_result.refutation_result}")
    print(f"  Original estimate: {refute_subset_result.initial_effect:.2f}")
    print(f"  New estimate (mean): {refute_subset_result.new_effect.mean():.2f}")
    print(f"  New estimate (std): {refute_subset_result.new_effect.std():.2f}")

    # 8. Predict Individual Causal Effects (CATE)
    # Let's create some new hypothetical customers to predict their CATE
    new_customers_data = pd.DataFrame({
        'age': [25, 40, 55],
        'income': [40000, 75000, 100000],
        'prior_purchases': [0, 2, 4],
        'marketing_campaign': [0, 0, 0], # Treatment value does not matter for CATE prediction
        'customer_lifetime_value': [0, 0, 0] # Outcome not needed for CATE prediction
    })
    
    # Preprocess new customer data using the *fitted* data processor
    processed_new_customers = pipeline.data_processor.transform(new_customers_data)

    cate_predictions = pipeline.causal_estimator.predict_individual_effect(processed_new_customers, 
                                                                         X_features=effect_modifiers,
                                                                         W_features=common_causes # W_features might not always be needed for prediction but good to pass for consistency
                                                                        )
    print("\nPredicted Conditional Average Treatment Effects (CATE) for new customers:")
    for i, (idx, row) in enumerate(new_customers_data.iterrows()):
        print(f"  Customer {idx+1} (Age: {int(row['age'])}, Income: {int(row['income'])}, Prior Purchases: {int(row['prior_purchases'])}): CATE = {cate_predictions[i]:.2f}")

    # 9. Save and Load the Pipeline for Deployment
    pipeline_save_path = f"{pipeline_name.lower().replace(' ', '_')}_pipeline.joblib"
    pipeline.save_pipeline(pipeline_save_path)
    logger.info(f"Pipeline saved to {pipeline_save_path}")

    loaded_pipeline = CausalPipeline.load_pipeline(pipeline_save_path)
    print(f"\nSuccessfully loaded pipeline '{loaded_pipeline.name}' for future use.")
    print(f"Loaded pipeline summary: {loaded_pipeline.get_summary()}")

    logger.info("Causal Insight Engine full example demo finished.")