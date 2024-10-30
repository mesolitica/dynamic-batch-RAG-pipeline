FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as base

RUN apt update
RUN apt install python3 python3-dev python3-pip -y
RUN apt install sudo -y
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/ubuntu --gecos "User" ubuntu
RUN usermod -aG sudo ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN pip3 install pip -U
RUN sudo apt update