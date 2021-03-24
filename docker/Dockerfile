FROM docker.io/rocm/tensorflow:rocm4.0.1-tf2.4-dev

RUN pip install pyyaml

# Replace 1000 with the UID of your user
# This is so the code in Docker runs as an unprivileged user instead of root.
# This avoids files being created as the root user in the mounted working directory.
RUN useradd -m -u 1000 user
