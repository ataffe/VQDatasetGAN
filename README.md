# SyntheticDataGen

This is an implementation of Latent Diffusion modified to generate a synthetic dataset of
surgical images and segmentation masks for surgical instruments. The goal is to be able to generate a synthetic dataset used to train a model for image segmentation using limited annotated data. I rearranged the code from
the [Latent Diffusion Repo](https://github.com/CompVis/latent-diffusion) and added the segmentation head from
[Big Dataset GAN](https://github.com/nv-tlabs/bigdatasetgan_code)