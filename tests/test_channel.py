import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import random
import threading
from unittest.mock import patch

from comms.channel import LossyChannel


def test_no_loss_all_messages_delivered():
    ch = LossyChannel(base_delay=0, jitter=0, loss_p=0)
    messages = [f"msg{i}" for i in range(5)]
    for msg in messages:
        ch.send(msg)
    time.sleep(0.01)
    received = ch.recv_ready()
    assert received == messages
    assert ch.recv_ready() == []


def test_loss_rate_deterministic():
    random.seed(0)
    ch = LossyChannel(base_delay=0, jitter=0, loss_p=0.5)
    messages = [f"msg{i}" for i in range(4)]
    for msg in messages:
        ch.send(msg)
    time.sleep(0.01)
    received = ch.recv_ready()
    assert received == ['msg0', 'msg2']


def test_latency_and_jitter_out_of_order():
    ch = LossyChannel(base_delay=0, jitter=0, loss_p=0)
    with patch("random.random", side_effect=[0.0, 0.0, 0.0]), \
         patch("random.gauss", side_effect=[0.02, 0.01, 0.03]):
        ch.send("m1")
        ch.send("m2")
        ch.send("m3")
    time.sleep(0.04)
    received = ch.recv_ready()
    assert received == ["m2", "m1", "m3"]


def test_queue_drains():
    ch = LossyChannel(base_delay=0.01, jitter=0, loss_p=0)
    ch.send("a")
    ch.send("b")
    time.sleep(0.02)
    assert ch.recv_ready() == ["a", "b"]
    assert ch.recv_ready() == []


def test_multiple_senders():
    ch = LossyChannel(base_delay=0, jitter=0, loss_p=0)

    def sender(name, count):
        for i in range(count):
            ch.send(f"{name}-{i}")

    t1 = threading.Thread(target=sender, args=("s1", 5))
    t2 = threading.Thread(target=sender, args=("s2", 5))
    t1.start(); t2.start()
    t1.join(); t2.join()
    time.sleep(0.01)
    received = ch.recv_ready()
    expected = {f"s1-{i}" for i in range(5)} | {f"s2-{i}" for i in range(5)}
    assert set(received) == expected
