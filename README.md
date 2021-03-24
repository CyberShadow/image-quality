Image Quality Experiment
========================

See [the blog post](https://blog.cy.md/2021/03/24/solving-xkcd-1683/) for a description.

Running the model
=================

Prerequisites:

- A [D compiler](https://dlang.org/download.html)
- ImageMagick
- Tensorflow 2.x (or Docker)

Setup:

- Make sure this repository is cloned recursively.
  Run `git submodule update --init --recursive` otherwise.

- Edit `docker-run.sh` according to the environment in which you run Python/TensorFlow/etc.
  If you have it installed natively on your host, replace its contents with `exec "$@"`.
  Running `./quality.sh check` should print `Python is OK`.

Scoring images:

- Run e.g.: `rdmd filescore.d xkcd/*.png`

Training the model
==================

1. Create a directory (or symbolic link pointing to one) called `images`, and populate it with at least 10000 images to use to generate the training data.

2. Create a directory (or symbolic link pointing to one) called `tests`, which will contain preprocessed images, edited versions, and metadata.

3. Run `rdmd gentests` to preprocess the test images and create edited versions.

4. Run `mkdata.sh` to generate the training data from the test images.

5. Run `./quality.sh fit` to fit the sample evaluator model.

6. Once satisfied, stop and rename the best model over `quality.h5`.

7. Run `./quality.sh fit_summarizer` to fit the summarizer model.

8. Once satisfied, stop and rename the best model over `summarizer.h5`.
