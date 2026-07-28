FROM apache/spark:3.5.6

USER root

# Install Python packages for PySpark & S3 / Kafka connectivity
RUN pip install --no-cache-dir \
    pyspark==3.5.6 \
    delta-spark==3.2.0 \
    pandas \
    pyarrow \
    boto3

USER 1001
