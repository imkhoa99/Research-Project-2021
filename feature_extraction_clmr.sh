#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --array=0-9
#SBATCH -o "slurm_%A_%a.out"

mkdir -p features
case $SLURM_ARRAY_TASK_ID in
   0)  GENRE=airport;;
   1)  GENRE=shopping_mall;;
   2)  GENRE=metro_station;;
   3)  GENRE=street_pedestrian;;
   4)  GENRE=public_square;;
   5)  GENRE=street_traffic;;
   6)  GENRE=tram;;
   7)  GENRE=bus;;
   8)  GENRE=metro;;
   9)  GENRE=park;;
esac

singularity exec --nv                \
  -B /scratch:/scratch               \
  -B $PWD/genres_our/audio/$GENRE:/input     \
  -B $PWD/features_our:/output         \
  -B $PWD/clmr_magnatagatune_mlp:/code/clmr_magnatagatune_mlp \
  --pwd /code                        \
  representations_clmr_latest.sif \
  python main.py
