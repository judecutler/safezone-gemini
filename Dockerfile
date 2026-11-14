FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY safezone/ safezone/
COPY web/ web/
COPY demo/ demo/
COPY fixtures/ fixtures/
ENV PORT=8080
CMD ["sh", "-c", "exec uvicorn safezone.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
