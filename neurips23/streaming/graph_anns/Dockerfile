FROM neurips23

RUN apt update && apt install -y rustc cargo
RUN pip install maturin
# copy everything from the grandparent directory into the container
COPY . /workspace
