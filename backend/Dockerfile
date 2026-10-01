FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 APP_ENV=production EDGELENS_DATA_DIR=/data TORCH_HOME=/data/weights
WORKDIR /app
ARG ML_EXTRA=onnx
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$ML_EXTRA" = "onnx" ] || [ "$ML_EXTRA" = "tflite" ]; then \
         pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
         && pip install --no-cache-dir -r requirements-ml.txt; \
       elif [ "$ML_EXTRA" != "none" ]; then exit 1; fi \
    && if [ "$ML_EXTRA" = "tflite" ]; then pip install --no-cache-dir -r requirements-tflite.txt; fi \
    && useradd --create-home --uid 10001 edgelens \
    && mkdir /data && chown edgelens:edgelens /data
COPY --chown=edgelens:edgelens app ./app
USER edgelens
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
