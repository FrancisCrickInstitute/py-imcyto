# py-imcyto
IMC segmentation in python.

pyimcyto is a python library for nuclear and cellular segmentation in imaging mass cytometry images.

The segmentation model uses a U-net++ archiecture with autoencoder-based anomaly detection to refine predictions, and is the same as the **deep-imcyto** segmentation model which is distributed as part of TRACERx-PHLEX -- a Nextflow-based multiplexed image analysis pipeline, however the implementation is more user friendly for those who wish to use the model itself directly in their own python code.

# Getting started

1. Download the trained deep-imcyto model weights from our Zenodo repository (https://doi.org/10.5281/zenodo.7573269)

2. Clone the pyimcyto repository

3. Setup the environment

'''console
conda create -n deepimcyto python=3.8 numpy pandas scipy matplotlib scikit-image cudatoolkit=11.8.0 scikit-learn -c conda-forge -c anaconda

<!-- conda install -c conda-forge cudatoolkit=11.8.0 scikit-learn -->

pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.* jupyter opencv-python cupy-cuda11x tqdm


'''

'''console
pip install jupyter tensorflow==2.12 opencv-python cupy-cuda11x tqdm
'''
# Credits

deep-imcyto/py-imyto is primarily developed by [Alastair Magness](mailto:alastair.magness@crick.ac.uk) at [The Francis Crick Institute](https://www.crick.ac.uk).