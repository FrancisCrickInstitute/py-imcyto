# py-imcyto
IMC segmentation in python.

pyimcyto is a python library for nuclear and cellular segmentation in imaging mass cytometry images.

The segmentation model uses a U-net++ archiecture with autoencoder-based anomaly detection to refine predictions, and is the same as the **deep-imcyto** segmentation model which is distributed as part of TRACERx-PHLEX -- a Nextflow-based multiplexed image analysis pipeline, however the implementation is more user friendly for those who wish to use the model itself directly in their own python code.

# Getting started

1. Download the trained deep-imcyto model weights from our Zenodo repository (https://doi.org/10.5281/zenodo.7573269)

2. Clone the pyimcyto repostitory




# Credits

deep-imcyto is primarily developed by [Alastair Magness](mailto:alastair.magness@crick.ac.uk) at [The Francis Crick Institute](https://www.crick.ac.uk).