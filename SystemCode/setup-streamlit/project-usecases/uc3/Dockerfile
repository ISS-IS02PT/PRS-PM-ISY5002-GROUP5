FROM python:3.8-slim-buster

# Our codebase folder
ENV BASEDIR=/deploy

# Setting WORKDIR for subsequent Docker commands
WORKDIR ${BASEDIR}

# Virtual Environment for Python
ENV VIRTUAL_ENV=${BASEDIR}/.venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ADD env_requirements.txt .
RUN pip install --no-cache-dir -r env_requirements.txt

ADD models ./models/
ADD constants_uc3.py .
ADD datapipeline_uc3.py .
ADD SessionState.py .
ADD st-usecase3.py .

EXPOSE 80
ENTRYPOINT ["streamlit", "run", "st-usecase3.py", "--server.port", "80"]