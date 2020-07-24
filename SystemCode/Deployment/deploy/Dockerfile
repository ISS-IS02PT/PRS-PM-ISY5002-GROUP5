FROM python:3.8-slim-buster

# Our codebase folder
ENV BASEDIR=/deploy

# Setting WORKDIR for subsequent Docker commands
WORKDIR ${BASEDIR}

# Virtual Environment for Python
ENV VIRTUAL_ENV=${BASEDIR}/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

ADD env_requirements.txt .
RUN pip install --no-cache-dir -r env_requirements.txt

ADD diabetes-knn-model.pkl .
ADD diabetes-scaler.pkl .

ADD flask_code.py .

EXPOSE 80
ENTRYPOINT ["python", "flask_code.py"]