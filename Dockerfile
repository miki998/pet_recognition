FROM gw000/keras
MAINTAINER Michael CHAN <miki998chan@gmail.com>

WORKDIR /home
COPY . .
RUN pip install -r requirements.txt && pip install jupyter
WORKDIR /home/DARK
RUN make
WORKDIR /home
EXPOSE 9999

STOPSIGNAL SIGTERM
