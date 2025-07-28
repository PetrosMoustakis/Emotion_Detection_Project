# Start with a standard Python 3.10 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your other files (app.py, model, templates/, etc.) into the container
COPY . .

# Expose the port Flask will run on
EXPOSE 7860

# The command to run your Flask application when the container starts
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]