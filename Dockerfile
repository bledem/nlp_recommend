FROM python:3.8

MAINTANER Your Name "betty.le.dem@gmail.com"

# We copy just the requirements.txt first to leverage Docker cache

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# COPY nlp_recommend .
# COPY models .

# ARG FLASK_ENV="developpment"
# ENV FLASK_ENV="${FLASK_ENV}" \
#     FLASK_APP="nlp_recommend/app/app" \
#     FLASK_SKIP_DOTENV="true" \
#     PYTHONUNBUFFERED="true" \
#     PYTHONPATH="." \
#     PATH="${PATH}:/home/python/.local/bin" \
#     USER="python"
    
# ENTRYPOINT [ "python" ]

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
