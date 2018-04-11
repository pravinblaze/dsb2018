# dsb2018
Project on Data Science Bowl 2018 conducted by kaggle

This project runs on python 3.6, pytorch, cuda 8.0

Make sure to also install any other packages that are imported at the top of the python notebook

### CONSIDER

* Custom immutable weights ?
* Visualize weights and heatmaps

ideas for backbone:
* train a cifar10 classifier
* prepare custom data set with iou thresholds
* if we're training the whole network end to end then, dont worry just build a good network

### TODO

* Complete datapipeline and produce masks for leaderboard submission

### IN PROGRESS

* Bounding box delta loss
* Mask loss

### DONE

* Vectorized end to end pipeline
* Classification loss for RPN
* Training target generator
* Write custom data loader for RPN
* Build and train better backbone with large enough perception field
* Explicit initialization on convolution layer weights
* Learning rate deacay
* Build backbone network, train till you get a good accuracy
* Also create random training data images to train against
* Based on masks, prepare box bound training crops from main images and pickle them in batches
* Analyze image shape and mask size characteristics
* Create unaltered data array batches and pickle them
