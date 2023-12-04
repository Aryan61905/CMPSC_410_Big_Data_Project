#!/bin/bash
#SBATCH --job-name=spark_job          # Job name
#SBATCH --nodes=8                     # Number of nodes to request
#SBATCH --ntasks-per-node=10           # Number of processes per node
#SBATCH --mem-per-cpu=8G                      # Memory per node
#SBATCH --time=3:00:00                # Maximum runtime in HH:MM:SS
#SBATCH --account=open

# Load necessary modules (if required)
module load anaconda3
source activate ds410_f23
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Run PySpark
# Record the start time
start_time=$(date +%s)
spark-submit --deploy-mode client --conf "spark.kryoserializer.buffer.max=2000m" --driver-memory 24g --executor-memory 26g Visuals.py

#python Lab11.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
