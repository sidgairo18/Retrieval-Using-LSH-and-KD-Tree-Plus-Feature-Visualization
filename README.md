# Neighbour Extraction

The dataset is stored on the scratch memory of the nodes listed in `nodes.txt`
`/scratch/bam_subset_2_0/`

The extracted features are stored at the locations mentioned in `features.txt`


Neighbour Extraction requires 1 GPU and ~30 CPUs to run optimally 

## How to run

1. Create a virtual environment using `virtualenv -p python3 venv`

2. Install pip packages listed in `Neighbour-Extraction/src/requirements.txt` using `pip3 install -r requirements.txt` inside the virtual environment

3. Run `Neighbour-Extraction/loadModules.sh` script to load cuda module if running on ADA

4. Use `Neighbour-Extraction/src/Label Dataset.ipynb` to store the dataset on scratch memory in the following format:

```
/scratch/bam_subset_2_0/<class_label>/<image_name>
```

5. Run `Neighbour-Extraction/src/getNeighbours.ipynb`
