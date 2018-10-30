# KNN

## How to run

- Copy `bam_subset` to `/scratch/` of ith node if not available
- `export $NODE='gnode<i>'`
- Allocate a node with GPU access by running `sinteractive -g 1 -A $USER -c 10 -w $NODE`
- Run `source loadModules.sh` to load the required modules
- Run the required scripts