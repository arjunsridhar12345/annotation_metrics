# annotation_metrics
# Environment Setup
First, create a new environment with a python version >= 3.9.

After doing this, clone the repo:

```
git clone https://github.com/arjunsridhar12345/annotation_metrics.git
```

And then `cd` to the directory where the repo was cloned and install the package and its dependencies

```
pip install -e .
```
Then `cd src/annotation_metrics`.

# Saving Correlation Plots
Run the command with the mouse id to save the correlation plots that the alignment app will use to the isilon.

```
python save_correlation_plots.py --mouseID 741137
```

# Saving Ethan's stim response metrics
Run the command to with the <strong>session id</strong> to save stim response metrics that the alignment app will use to the isilon.

```
python save_stim_metrics_for_alignment.py --sessionID 741137_2024-10-10
```
