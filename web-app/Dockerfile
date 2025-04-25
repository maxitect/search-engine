FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY search_engine.py .

# Copy the Streamlit configuration
COPY .streamlit /app/.streamlit

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "search_engine.py", "--server.address=0.0.0.0"]