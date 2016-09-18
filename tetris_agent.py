import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import socket
import sys
import os
import datetime
import time

def main():
    self.broadcast_port = 8080
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    self.sock.bind(('', 0))

    index = 1
    while True:
        self.sock.sendto('...' + index, ("<broadcast>", self.broadcast_port))
        index += 1
        sys.stdout.flush()

if __name__ == "__main__":
    main()
