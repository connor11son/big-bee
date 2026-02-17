FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /module

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY model_cache/ ./model_cache/
COPY pyproject.toml uv.lock ./
COPY segment_wing.py ./
COPY PythonScriptWrapper.py ./
COPY BQ_run_module.py ./

# Install dependencies from lock file
RUN uv sync --frozen --no-dev

# Create directories for input/output mounting
RUN mkdir -p /input /output


# Replace the following line with your {ModuleName}.xml
COPY bigbeedetector.xml /module/bigbeedetector.xml

ENV PATH /module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

CMD ["--help"]
