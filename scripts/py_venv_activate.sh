
# wb_command
module load connectomeworkbench


# create new python env, ENV is your env name
virtualenv --no-download ENV

salloc --account=def-bashivan --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=4:00:00
module load python/3.10
module load scipy-stack
# activate teh environment
source nilearn/bin/activate


pip install --no-index --upgrade pip