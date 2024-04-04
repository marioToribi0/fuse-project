FROM python:3.10.2

WORKDIR /app

RUN pip install poetry

# copy all files
COPY . . 

RUN poetry install

EXPOSE 80

ENTRYPOINT ["poetry", "run", "streamlit", "run"]

CMD ["main.py"]