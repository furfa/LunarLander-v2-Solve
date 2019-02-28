#! /bin/bash

1="OpenAi";


ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *"$1"* ]]; then
   source activate $1
else 
   echo "INSTALLING"
   conda env create -f environment.yml
   source activate $1
fi;

echo Ezz;
