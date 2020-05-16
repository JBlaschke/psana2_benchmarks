#!/usr/bin/env python
# -*- coding: utf-8 -*-


from time import sleep


from benchmarking import Event, EventLogger


def test_timers(n):

    e1 = Event()
    sleep(n)  # sleep for n seconds
    e2 = Event()

    return e1, e2


if __name__ == "__main__":

    e1, e2 = test_timers(2)
    print("Testing event timers:")
    print(f"e1 = {e1.timestamp}\ne2 = {e2.timestamp}")
