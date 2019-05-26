FROM ubuntu



RUN apt update ; apt-get -y install locales ; apt install -y curl git ; \
    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Set the locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY run.sh ./

ENTRYPOINT bash ./run.sh

