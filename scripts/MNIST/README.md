# Scripts for MNIST point cloud

This folder currently contains simple HMill classifier model and a work-in-progress Kingma M2 model.

*Important note for M2 model: The categorical distribution only works for labels from 1 to c, where c is the number of categories. MNIST has labels from 0-9 and therefore needs to be encoded to 1-10. For HMill classifier this is a not a problem thanks to the making of onehot encoding through `Flux.onehotbatch(y, classes).*

## To-do

- [ ] add Triplet regularization to HMill classifier
- [ ] add semi-supervised approach to HMill classifier through clustering in the latent space for data with unknown labels (*Note: Implement something like only use the datapoint if we are somewhat sure that the label inferred is right.*)

