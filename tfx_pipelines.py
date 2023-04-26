# -*- coding: utf-8 -*-
"""tfx_pipelines_airline.ipynb

# **Submission 2 Proyek Pengembangan dan Pengoperasian Sistem Machine Learning**

## **Data Preparation (Kaggle)**

### **Dataset Summary**

This dataset contains an [US Airline Passenger Satisfaction Survey](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
"""


"""## **Library**"""

# import library
import os
import pandas as pd
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


"""## **Data Loading**"""

# initialize path
pd.reset_option('^display.', silent=True)

DATA_PATH = 'data'

# let's see dataframe
df = pd.read_csv(os.path.join(DATA_PATH, 'airline_passenger_satisfaction.csv'))
df.head()

"""**DESCRIPTION OF DATA**

There is the following information about the passengers of some airline:

*   ***Gender***: Gender of the passengers (Female, Male)
*   ***Customer Type***: The customer type (Loyal customer, disloyal customer)
*   ***Age***: The actual age of the passengers
*   ***Type of Travel***: Purpose of the flight of the passengers (Personal Travel, Business Travel)
*   ***Class***: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
*   ***Flight distance***: The flight distance of this journey
*   ***Inflight wifi service***: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
*   ***Departure/Arrival time convenient***: Satisfaction level of Departure/Arrival time convenient
*   ***Ease of Online booking***: Satisfaction level of online booking
*   ***Gate location***: Satisfaction level of Gate location
*   ***Food and drink***: Satisfaction level of Food and drink
*   ***Online boarding***: Satisfaction level of online boarding
*   ***Seat comfort***: Satisfaction level of Seat comfort
*   ***Inflight entertainment***: Satisfaction level of inflight entertainment
*   ***On-board service***: Satisfaction level of On-board service
*   ***Leg room service***: Satisfaction level of Leg room service
*   ***Baggage handling***: Satisfaction level of baggage handling
*   ***Check-in service***: Satisfaction level of Check-in service
*   ***Inflight service***: Satisfaction level of inflight service
*   ***Cleanliness***: Satisfaction level of Cleanliness
*   ***Departure Delay in Minutes***: Minutes delayed when departure
*   ***Arrival Delay in Minutes***: Minutes delayed when Arrival
*   ***Satisfaction***: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)
"""


"""## **Set Pipeline Variable**"""

# pipeline name
PIPELINE_NAME = 'airline-satisfaction-pipeline'

# pipeline inputs
DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/airline_satisfaction_transform.py'
TUNER_MODULE_FILE = 'modules/airline_satisfaction_tuner.py'
TRAINER_MODULE_FILE = 'modules/airline_satisfaction_trainer.py'

# pipeline outputs
OUTPUT_BASE = 'outputs'

serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')


"""## **ML Pipelines (Pipeline Orchestrator)**"""

# build TFX component
def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """Init local pipeline

    Args:
        components (dict): tfx components
        pipeline_root (Text): path to pipeline directory

    Returns:
        pipeline.Pipeline: apache beam pipeline orchestration
    """
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        '--direct_running_mode=multi_processing'
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        '----direct_num_workers=0'
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )


# running pipelines
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules import components

    components = components.init_components({
        'data_dir': DATA_ROOT,
        'transform_module': TRANSFORM_MODULE_FILE,
        'tuner_module': TUNER_MODULE_FILE,
        'training_module': TRAINER_MODULE_FILE,
        'training_steps': 5000,
        'eval_steps': 1000,
        'serving_model_dir': serving_model_dir
    })

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
