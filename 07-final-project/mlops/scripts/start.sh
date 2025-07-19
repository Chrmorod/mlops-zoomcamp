#!/bin/bash
export STOCK=${STOCK:-MSFT}
export YEAR_BACK=${YEAR_BACK:-3}

PROJECT_NAME=mlops \
  MAGE_CODE_PATH=/home/src \
  SMTP_EMAIL=$SMTP_EMAIL \
  SMTP_PASSWORD=$SMTP_PASSWORD \
  STOCK=$STOCK \
  YEAR_BACK=$YEAR_BACK \
  docker compose up

#STOCK=AAPL YEAR_BACK=5 ./start.sh