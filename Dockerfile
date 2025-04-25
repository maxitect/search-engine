# Start off with an image that has python version 3.11 installed
FROM continuumio/miniconda3

# Create a directory and move into it
WORKDIR /code


# Copy the requirements in first, so that when we rebuild the image, it
# can use the same layer (essentially the cache), and skip this part
COPY environment.yml .

# Install all the requirements. Will also be skipped if not changes
# have been made
RUN echo "----> Creating Conda environment 'ss-env' from environment.yml..." && \
    conda env create -v -f environment.yml

# Copy inside the rest of our app that 
COPY . .

RUN echo "----> Copied application code into /code."

# This just documents and makes it visible in the command line, which
# port is supposed to be exposed
EXPOSE 8501

# As long as the command line below is running, the container will
# stay alive. If this command crashes or fails, the container dies
CMD ["conda", "run", "-n", "ss-env", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]