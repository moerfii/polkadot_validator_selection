# Polkadot Machine Learning Alternative to Sequential Phragmén
As Polkadot is growing, it becomes evident that a need for scaling of the staking system emerges. The current implementation, the sequential Phragmén, has its inherent limitations due to its dependent time complexity. The limitations are reflected in the maximum number of nominators (22'500) that are allowed to participate, each of them selecting a maximum of 16 validators. The goal was to deliver a machine learning approach that can address the problem by providing equal or better solutions, while scaling better than the current implementation.
## Instructions

1. Create new virtual environment ```python3 -m venv venv```
2. Activate virtual environment ```source venv/bin/activate```
3. Install dependencies ```make install```
4. Fill out the config file 'config.json'. 
5. ```cd sequential_phragmen_local``` and ```cargo build```
6. Return back to root directory and run the program ```python3 main.py```




## config.json options

```
{
  "username": "username",           supply username to postgres (create account in Florians DB)
  "password": "password",           supply password to user
  "database": "database",           supply url to database

  "era": 1033,                      Define era to predict



  "intermediate_results_path": "./data_collection/data/intermediate_results/",           # set path


################## Active Set Predictor ##############################
  "model_1_path": "./data_collection/data/processed_data/model_1_data",                  # set path
  "model_1": "lgbm_classifier",                                                          # set model
  "features_1": [                                                                        # add or remove features
    "overall_total_bond",
    "overall_proportional_bond",
    "nominator_count",
    "elected_previous_era",
    "era"
  ],
  "target_1": "elected_current_era",                                                      # define target

################# Expected Sum Predictor ############################

  "model_2_path": "./data_collection/data/processed_data/model_2_data",
  "model_2": "lgbm_model_2",
  "model_2_load": "./models/trained_models/lgbm.pkl",
  "scaler_2_load": "./models/trained_models/lgbm_scaler.pkl",
  "features_2": [
    "validator",
    "proportional_bond",
    "total_bond",
    "validator_frequency_current_era",
    "probability_of_selection",
    "validator_centrality",
    "era"
  ],
  "target_2": "solution_bond",

################# Stake Distribution Predictor #########################

    "model_3_path": "./data_collection/data/processed_data/model_3_data",
    "model_3": "lgbm_model_3",
  "features_3": [
    "nominator",
    "validator",
    "proportional_bond",
    "total_bond",
    "overall_total_bond",
    "overall_proportional_bond",
    "era",
    "number_of_validators",
    "validator_frequency_current_era",
    "average_proportional_bond",
    "average_total_bond",
    "nominator_index",
    "validator_index",
    "nominator_centrality",
    "validator_centrality",
    "probability_of_selection",
    "expected_sum_stake"
  ],
    "target_3": "solution_bond",
    "compare": "./data_collection/data/calculated_solutions_data/",
    "plot": null,
  "save": null,

###########################################################################
Only necessary if you prefer to pull data via storage queries.
  "node": {
    "url": "wss://rpc.polkadot.io",
    "ss58_format": 0,
    "type_registry_preset": "polkadot"
  },

}
```