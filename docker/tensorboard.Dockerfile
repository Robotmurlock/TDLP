FROM python:3.9-slim

# Install TensorBoard
RUN pip install --no-cache-dir tensorboard

# Create log directory
RUN mkdir -p /tb_logs
WORKDIR /tb_logs

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["tensorboard", "--host", "0.0.0.0", "--port", "6006", "--logdir", "/tb_logs", "--load_fast=false", "--reload_multifile", "True"]
