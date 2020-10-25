FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
MAINTAINER Samson Lee <samsonleegh@gmail.com>

# install build utilities
# RUN apt-get update && \
# 	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
# RUN python3 --version
# RUN pip3 --version

# set the working directory for containers
ARG WORK_DIR="/usr/src/convai_smile"
WORKDIR $WORK_DIR

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add a line here to update the conda environment using the conda.yml. 
# Remember to specify that the environment to update is 'polyaxon'.
# RUN conda env update -n base --file conda.yml
# RUN rm conda.yml

# Add lines here to copy over your src folder and 
# any other files you need in the image (like the saved model).
COPY ./src $WORK_DIR/src
COPY ./cache_dir $WORK_DIR/cache_dir
COPY ./saved_model $WORK_DIR/saved_model

# RUN chown -R 1000450000:0 $WORK_DIR

# USER $USER

EXPOSE 8000

# Add a line here to run your app
# CMD ["uvicorn", "src.main:app"]
ENTRYPOINT ["uvicorn"]
CMD ["src.main:app", "--host", "0.0.0.0"]

# docker build /Users/samsonlee/Documents/aisg/projects/chatbot/convai_smile -t convai_smile
# docker run -p 8000:8000 convai_smile