FROM huggingface/transformers-pytorch-gpu:latest

# We copy just the requirements.txt first to leverage Docker cache

WORKDIR /app

# Update pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip install --upgrade pip setuptools wheel

RUN pip3 install numpy
RUN pip3 install pandas 
RUN pip3 install scipy
COPY ./requirements_docker.txt requirements_docker.txt

RUN pip install --no-cache-dir -r requirements_docker.txt

WORKDIR /app/nlp_recommend

COPY nlp_recommend ./nlp_recommend

ARG FLASK_ENV="developpment"

ENV FLASK_ENV="${FLASK_ENV}" \
    FLASK_APP="nlp_recommend/app/webapp_light" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \ 
    PYTHON_PATH="/app/nlp_recommend"

# CMD ["flask", "run", "--host=0.0.0.0", "-p", "5000"]
CMD ["python3", "nlp_recommend/app/webapp_light.py", "--data_path", "/training"]
