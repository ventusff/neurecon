# put project directory into PYTHONPATH

# source $HOME/.condabashrc
# conda activate nerf

DIR="$(pwd)"
export PYTHONPATH="${DIR}":$PYTHONPATH
echo "added $DIR to PYTHONPATH"
