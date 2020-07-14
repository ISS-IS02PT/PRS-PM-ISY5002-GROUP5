## 1. Run the following commands from the Venv Python / Conda environment of Jupyter notebook

```bash
# Install jupytext from pip
$ pip install jupytext
```
or
```bash
# Install jupytext from conda
$ conda install -c conda-forge jupytext
```

## 2. Update Jupytext's contents manager
Append
```code
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
```
to your .jupyter/jupyter_notebook_config.py file (generate a Jupyter config, if you don't have one yet, with jupyter notebook --generate-config)

## 3. Run a build step
```bash
$ jupyter lab build
```

## 4. Pair the notebook
Follow the step here:
https://jupytext.readthedocs.io/en/latest/paired-notebooks.html