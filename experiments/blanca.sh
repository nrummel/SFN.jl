#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --job-name=rsfn_efficiency
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --no-requeue
#SBATCH --partition=blanca-appm
#SBATCH --account=blanca-appm
#SBATCH --qos=blanca-appm-student

ROOT=/projects/cosi1728/R-SFN/experiments

module purge
module julia/1.6.6

julia --project=$ROOT --threads=10 -- $ROOT/efficiency.jl

cp efficiency_data.jld2 $ROOT/results/
