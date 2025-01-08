# ============ Compile Nextjs App ============
# Build nextjs app and copy to alpine image
FROM node:23 as build

# Set the working directory
WORKDIR /app

# Copy package.json and install dependencies
COPY --chown=node:node ./web .
RUN chmod +x node_modules/.bin/next
RUN npm install

# Build the app
RUN npm run build

# ============ Build Image ============
FROM alpine:3.19

# This hack is widely applied to avoid python printing issues in docker containers.
# See: https://github.com/Docker-Hub-frolvlad/docker-alpine-python3/pull/13
ENV PYTHONUNBUFFERED=1

RUN echo "**** install Python ****" && \
    apk add --no-cache python3 && \
    if [ ! -e /usr/bin/python ]; then ln -sf python3 /usr/bin/python ; fi && \
    \
    echo "**** install pip ****" && \
    rm /usr/lib/python3.11/EXTERNALLY-MANAGED && \
    python -m ensurepip && \
    rm -r /usr/lib/python*/ensurepip && \
    if [ ! -e /usr/bin/pip ]; then ln -s pip3 /usr/bin/pip ; fi && \
    pip install --no-cache --upgrade pip setuptools wheel


# Set the working directory
WORKDIR /app

# Copy local code to the /app folder
COPY requirements.txt .
COPY .env .
COPY *.py .

# Copy the build files from the build image
COPY --from=build ../ui ./ui

# Install the dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Expose the port the app runs on
EXPOSE 80

# Serve the app
CMD ["python", "chat_with_netty.py"]