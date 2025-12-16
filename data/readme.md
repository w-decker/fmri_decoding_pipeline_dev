# Simulate fMRI data for decoding pipeline

This code simulates fMRI volumes for multivariate decoding for a single or multi-subject regime.

```{mermaid}
---
title: Flow
---
flowchart LR
    subgraph 1
    direction TB
    A(zsh make_stimufunc.sh)
    A --> A1(**Must** have FreeSurfer installed)
    A1 --> A2(Uses `optseq2` to generate 10 stimulus timing files. The first is used to simulate data.)
    end
    
    subgraph 2
    direction TB
    B(python make_data.py) --> B1(Define data dimensions)
    B1 --> B2(Define noise parameters)
    B2 --> B3(Define signal and timepoints of interest)
    B3 --> B4(Linearly combine noise and signal)
    end

    subgraph 3
    direction TB
    C(python first_level.py)
    end

    1 --> 2
    2 --> 3
    
    style A fill:#e09,color:#000
    style B fill:#94b4e0,color:#000
    style C fill:#9CB394,color:#000
```

# Generating stimulus timing files
To generate stimulus timing files, run the following code.

```zsh
zsh make_stimfunc.sh
```
Results will be saved in [`optseq2/`](/data/optseq2/). `stimfunc-001.par` is used in `make_data.py` by default.

# Run simulation process

Data are simulated using `brainiak.utils.fmrisim`. See [here](https://brainiak.org/examples/fmrisim_multivariate_example.html) for a tutorial.

To simulate fMRI data for a single subject, run the following code.

```zsh
conda create -f env.yml
conda activate decoding
python make_data.py
```
There are _**two**_ arguments required to run this script: `--directory` (where to save the simulated data) and `--seed` (a random seed). Defaults for the rest of the simulation process are embedded in the code in the `parse_args()` function.