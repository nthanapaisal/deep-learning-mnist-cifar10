### env set up:
    source ~/miniconda3/bin/activate &&
    conda create -n ml python=3.11 &&
    conda activate ml &&
    conda install pandas && 
    conda install numpy &&
    pip install torch torchvision             # instead of conda install since it won't work with my m4 chip


### specs:
    Macbook M4 Pro (2024)