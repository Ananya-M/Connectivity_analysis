# Open-data connectivity (ds002680)

The SOMA connectivity analysis applied to OpenNeuro **ds002680** (Delorme et al., go/no-go rapid animal categorisation; 31-channel EEG, 1000 Hz, BIDS/EEGLAB `.set`).

- `notebooks/01_pre_post_stimulus.ipynb`: part 1, pre vs post-stimulus PLI/wPLI on homologous pairs and ITC per hemisphere, for targets and distractors.
- `notebooks/02_target_vs_distractor.ipynb`: part 2, all-pairs connectivity matrices, target vs distractor, pre vs post, alpha and beta.

## Setup

```powershell
$env:UV_PROJECT_ENVIRONMENT = "conn_open"; uv sync
.\conn_open\Scripts\python.exe -m ipykernel install --user --name conn_open --display-name conn_open
```

Pick the **conn_open** kernel in the notebooks. Data location is set in `config/paths.yaml` (`bids_root`); recordings are found at any folder depth, so a full BIDS download (`sub-*/ses-*/eeg/...`) works as is.
