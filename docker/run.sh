#!/usr/bin/env bash

ENV_PATH=${ENV_PATH:-.}
APP_PATH=${APP_PATH:-.}


export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

source ${ENV_PATH}

cd ${APP_PATH}

gunicorn -c gunicorn_config.py wsgi:application

