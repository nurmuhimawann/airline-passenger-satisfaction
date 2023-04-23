# -*- coding: utf-8 -*-
"""tfx_pipelines.ipynb

# **Submission 2 Proyek Pengembangan dan Pengoperasian Sistem Machine Learning**

## **Data Preparation (Kaggle)**

### **Dataset Summary**

This dataset corresponds to an augmented version of the [**Electrical Grid Stability Simulated
Dataset**](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+#),
created by Vadim Arzamasov (Karlsruher Institut für Technologie, Karlsruhe, Germany)
and donated to the [University of California (UCI) Machine Learning
Repository](https://archive.ics.uci.edu/ml/index.php)


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

"""The dataset chosen for this machine learning exercise has a synthetic nature and contains results
from simulations of grid stability for a reference 4-node star network, as described in 1.2.

The original dataset contains 10,000 observations. As the reference grid is symetric, the dataset
can be augmented in 3! (3 factorial) times, or 6 times, representing a permutation of the three
consumers occupying three consumer nodes. The augmented version has then **60,000 observations**.
It also contains **12 primary predictive features** and two dependent variables.

**Predictive features:**
<ol>
    <li>'tau1' to 'tau4': the reaction time of each network participant, a real value within
    the range 0.5 to 10 ('tau1' corresponds to the supplier node, 'tau2' to 'tau4' to
    the consumer nodes);</li>
    <li>'p1' to 'p4': nominal power produced (positive) or consumed (negative) by each network
    participant, a real value within the range -2.0 to -0.5 for consumers ('p2' to 'p4').
    As the total power consumed equals the total power generated,
    p1 (supplier node) = - (p2 + p3 + p4);</li>
    <li>'g1' to 'g4': price elasticity coefficient for each network participant, a real value
    within the range 0.05 to 1.00 ('g1' corresponds to the supplier node, 'g2' to 'g4' to
    the consumer nodes; 'g' stands for 'gamma');</li>
</ol>

**Dependent variables:**
<ol>
    <li>'stab': the maximum real part of the characteristic differentia equation root
    (if positive, the system is linearly unstable; if negative, linearly stable);</li>
    <li>'stabf': a categorical (binary) label ('stable' or 'unstable').</li>
</ol>

As there is a direct relationship between 'stab' and 'stabf'
('stabf' = 'stable' if 'stab' <= 0, 'unstable' otherwise), 'stab' will be dropped
and 'stabf' will remain as the sole dependent variable.
"""

# let's see dataframe
df = pd.read_csv(os.path.join(DATA_PATH, 'smart_grid_stability.csv'))
df.head()

"""## **Set Pipeline Variable**"""

# pipeline name
PIPELINE_NAME = "smart-grid-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/smart_grid_transform.py"
TUNER_MODULE_FILE = "modules/smart_grid_tuner.py"
TRAINER_MODULE_FILE = "modules/smart_grid_trainer.py"

# pipeline outputs
OUTPUT_BASE = "output"

serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")

"""## **ML Pipelines (Pipeline Orchestrator)**"""

# build TFX component
def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0"
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
