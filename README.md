This repository generates results for our paper, where we analyzed the differences between left and right profiles for sea turtles. We analyzed three different species (loggerheads, greens and hawksbills) with the uniform conclusion is that there is a significant similarity between opposite profile in all three species. The main conclusion of this observation is that biologists should used both profiles for identifying individuals and not only the same profile as the current practise goes.

## Installation

To install the repository, download it first. This can be either done manually or when git is installed by
```script
git clone https://github.com/sadda/sides-matching.git
cd sides-matching
```

Optionally create a virtual environement. To install the required packages, open the console and run

```script
pip install -r requirements.txt
```

## Usage

The code is divided into the notebooks:

- The [first notebook](notebooks/compute_features.ipynb) downloads the three datasets and extracts features from all images.
- The [second notebook](notebooks/matching.ipynb) uses the extracted features to compute similarities between images. Based on these similarities, it predicts which individual turtles are depicted in images. This predictions are then use to conclude that there is a significant similarity between the left and right profiles of individual turtles.
