FROM python:3.6

WORKDIR /app

RUN curl https://cmake.org/files/v3.10/cmake-3.10.3-Linux-x86_64.sh -o /tmp/curl-install.sh \
      && chmod u+x /tmp/curl-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/curl-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/curl-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir osqp

COPY . .

CMD ["python", "./src/ethylene_oxide_ihmpc.py"]