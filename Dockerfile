FROM python:3.11-slim

WORKDIR /


# Copy your handler file and requirements
COPY llm.py /
COPY ir.py /
COPY rp_handler_ask_a_phil.py /
COPY requirements.txt /
COPY aristotle.json /
COPY confucius.json /

# Install dependencies
RUN pip install -r requirements.txt

# Start the container
CMD ["python3", "-u", "rp_handler_ask_a_phil.py"]