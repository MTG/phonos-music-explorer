FROM ubuntu:14.04
MAINTAINER Giuseppe Bandiera "giuseppe.bandiera@live.com"
RUN apt-get update && apt-get install -y \
    python-pip \
    python-dev \
    build-essential \
    libmp3lame-dev \
    python-gi \
    python-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    vlc \
    browser-plugin-vlc
COPY requirements.txt /usr/local/empd/requirements.txt
WORKDIR /usr/local/empd
RUN pip install -r requirements.txt
COPY run.py /usr/local/empd
COPY app/ /usr/local/empd/app
ENTRYPOINT ["python"]
CMD ["run.py"]
