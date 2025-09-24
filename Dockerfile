# Use a Python base image and install Java for PySpark
FROM python:3.10-slim-bullseye

# Install Java Development Kit (JDK) for PySpark and other utilities
RUN apt-get update && apt-get install -y openjdk-11-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME dynamically to work across different architectures
RUN export ARCH=$(dpkg --print-architecture) && \
    ln -sf /usr/lib/jvm/java-11-openjdk-${ARCH} /usr/lib/jvm/java-11-openjdk

ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Force IPv4 for Java/Spark to avoid IPv6 issues in Cloud Run
ENV JAVA_OPTS="-Djava.net.preferIPv4Stack=true -Djava.net.preferIPv4Addresses=true"
ENV SPARK_SUBMIT_OPTS="-Djava.net.preferIPv4Stack=true"

# Set environment variables for Spark (needed by PySpark)
# SPARK_HOME points to the PySpark installation directory for Python 3.10
ENV SPARK_HOME="/usr/local/lib/python3.10/site-packages/pyspark"
ENV PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/jars"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY monthly_transactions.csv /app/
COPY src/ /app/src/
COPY main.py /app/

RUN mkdir -p /app/models
COPY models/ /app/models/

# Verify that critical files are present
RUN ls -la /app/ && ls -la /app/models/ && ls -la /app/src/ && \
    test -f /app/monthly_transactions.csv && \
    test -d /app/models/forecasting_pipeline_model && \
    echo "✓ All required files are present in the container"

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "main.py"]
