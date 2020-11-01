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

ADD uber_pickups.py .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "uber_pickups.py"]