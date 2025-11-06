### env set up:
    source ~/miniconda3/bin/activate &&
    conda create -n ml python=3.11 &&
    conda activate ml &&
    conda install pandas && 
    conda install numpy &&
    pip install torch torchvision                   # Instead of conda install since it won't work with my m4 chip


### specs:
    Macbook M4 Pro (2024)


# How to run:
    Files Part 1:
        0. preprocessing.py
        1. mlps_shallow.py
        2. mlps_medium.py
        3. mlps_deep.py
        4. main.py
    
        >    python3 main.py                         # This will run all of them together and produce nn_final_report.py

    Files Part 2:
        1. cnn_baseline.py
        2. cnn_enhanced.py
        3. cnn_deep.py
        4. compile_output.py

        > python3 filename.py                         # I ran these files separately since it could take a while. This will produce..
                                                      # cnn_baseline_report.json, cnn_enhanced_report.json cnn_deep_report.json
        > python3 compile_output.py                   # this will produce the final result (summary_tale.csv) with table that shown in project3_report.pdf
