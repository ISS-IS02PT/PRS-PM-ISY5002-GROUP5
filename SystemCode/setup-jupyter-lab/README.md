## 1. Create a new virtual environment, with Python3

```bash
# Assuming you are at the SystemCode folder
$ pwd
/Users/kenly/Documents/Work/ISS-IS02PT/PRS-PM-ISY5002-GROUP5/SystemCode

# Create the new environment named 'sandbox' in the folder 'venv'
$ python3 -m venv venv/sandbox
```

## 2. Activate the new virtual environment
```bash
# Execute this command
$ source venv/sandbox/bin/activate

# You should now see the name 'sandbox' in front
(sandbox) $ 
```

## 3. Prepare pip
```bash
(sandbox) $ pip install —-upgrade pip

# This allows you to remove pip package easily
(sandbox) $ pip install pip-autoremove

# This integrates with notebook to install pip package right from notebook cell. Just do '%pip install numpy' in a cell
(sandbox) $ pip install pip-magic
```

## 4. Install Jupyter Lab
```bash
(sandbox) $ pip install jupyterlab
```

## 5. Start Juputer Lab
```bash
(sandbox) $ jupyter lab --port=9999
```