# Stroke Project- Dr. Eickhoff

This repository contains the public code for our paper, [Delirium detection using wearable sensors and machine learning in patients with intracerebral hemorrhage](<https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1135472/full>), published in *Frontiers in Neurology*.

## Installation Steps:

1. Clone this directory
2. Navigate to root of directory
3. Build Docker image with the following command:
 
 ```bash
 docker build -t delirium-stroke:latest .
 ```

 4. Run Docker container with the following command:

```bash
docker run -it --rm \
-v <path to p_eickhoff_stroke>:/home/p_eickhoff_stroke \
-v <path to raw stroke data>:/home/stroke_data \
delirium-stroke:latest
```

Executing this command runs ```experiments.py``` with the default parameters. To run with different parameter settings, manually add
the command function. For instance, to run a Baseline model on next-day prediction without actigraph data, use the following command:

```bash
docker run -it --rm -v <path to p_eickhoff_stroke>:/home/p_eichoff_stroke delirium-stroke:latest -v <path to raw stroke data>:/home/stroke_data python experiments.py --no_actigraph --next_day --classifier Baseline
```

Here is an example of how I run the default command on my Windows machine (hence the backward slashes in my raw paths):
```bash
docker run -it --rm \
-v C:\Users\abdul\Desktop\p_eickhoff-stroke:/home/p_eickhoff_stroke \
-v C:\Users\abdul\Desktop\stroke_data:/home/stroke_data \
delirium-stroke:latest
```

The ```stroke_data``` folder contains the raw actigraph and ICH data for the 40 patients in this study. You will need access to the shared Google Drive to download it (or contact me at abdullah_ahmed@brown.edu)





