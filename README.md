# ev-wgsm
This code describes the basic analysis workflow in  "P. Parkkila et al.: Surface-sensitive waveguide scattering microscopy of single extracellular vesicles reveals content and biomarker heterogeneity". It has three notebooks:
## [preprocessing.ipynb](preprocessing.ipynb)
* Forming scattering frames before and after particle additions
* Adding scattering frames from glycerol/iodixanol changes
* Aligning the frames
* Forming fluorescence frames before and after antibody incubations
* Forming time-resolved antibody binding frames in both scattering and fluorescence modes

## [analysis.ipynb](analysis.ipynb)
* Loading the frames formed in [preprocessing.ipynb](preprocessing.ipynb)
* Detecting particles in scattering and fluorescence frames
* Intensity calculations for the particles detected
* Calculating interparticle distances (for overlap detection)
* Aligning scattering and fluorescence frames by translation, rotation and shear
* Aligning time-resolved fluorescence frames
* Calculating intensities from time-resolved fluorescence and scattering frames
* Calculating and normalizing intensities from iodixanol/glycerol exchanges
* Fitting based on Mie scattering models
* Calculating membrane thickness changes upon antibody binding

## [plot.ipynb](plot.ipynb)
* Example plot of effective RI vs. diameter

## Downloading and using the example data
The example dataset can be downloaded from [Zenodo](https://zenodo.org/records/12529089?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTcxOTMyNTI4NiwiZXhwIjoxNzUwODA5NTk5fQ.eyJpZCI6IjE5ZWY5YzZhLTkwZGUtNGQyZi1hOGIyLWRkYzU0Y2NkNDk2OCIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZDhiNjgwYTRiZGRjODEzMDZkODU5YmI1YWNjYTkzMCJ9.gO8ivnAidHvTd3TrDYABZWjuTnTFrHcvTwgRp7skhlaHWZ6WzfL8_kXMLFf1NnTJ1pmIUU5-TWQv_m_k3guj2g). Place the data in the folder ev-wgsm/example-data. Processed data is placed in ev-wgsm/example-data/processed. The data corresponds to a platelet EV (PEV) experiment used in the manuscript figures 3a,c,d, 4a,b,d and 5a,b.

## Requirements
Conda environment file ev_wgsm.yml is provided in the root folder. Use `conda env create -f ev_wgsm.yml` in the parent folder of your Anaconda distribution. 
```
conda activate ev_wgsm
git clone https://github.com/hooklab-chalmers/ev-wgsm.git
```
* Version tested: 1.0.0 (Windows 11, 23H2)
* Software: Anaconda 2024.02. 
* Hardware: Standard computer
* Installation time: ~10 min
* Run time: ~1 hour with Mie scattering-based fitting