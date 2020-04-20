FROM python:3.7-slim-stretch

RUN echo "|--> Install gunicorn" \
    && pip install gunicorn

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./gunicorn_conf.py /gunicorn_conf.py

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

COPY ./start-reload.sh /start-reload.sh
RUN chmod +x /start-reload.sh

COPY ./satransformers /app/satransformers
COPY ./app /app
WORKDIR /app/

ENV PYTHONPATH=/app

EXPOSE 80

# Run the start script, it will check for an /app/prestart.sh script (e.g. for migrations)
# And then will start Gunicorn with Uvicorn
CMD ["/start.sh"]