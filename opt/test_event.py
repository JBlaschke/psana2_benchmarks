#!/usr/bin/env python
# -*- coding: utf-8 -*-


from time import sleep

from benchmarking import Event, EventLogger, event_here



def log_timer(e):
    EventLogger().add(e)


def test_timers(n):

    e1 = Event(label="l 1")
    sleep(n)  # sleep for n seconds
    e2 = Event(label="l 2")

    return e1, e2



def test_logger(n):
    EventLogger().clear()

    e1 = Event(label="l 3")
    sleep(n)  # sleep for n seconds
    e2 = Event(label="l 4")

    log_timer(e1)
    log_timer(e2)



def test_inplace_logger(n):
    EventLogger().clear()

    event_here(label="l 5")
    sleep(n)  # sleep for n seconds
    event_here(label="l 6")



if __name__ == "__main__":

    e1, e2 = test_timers(1)
    print("Testing event timers:")
    print(f"e1 = {e1.label}@{e1.timestamp}\ne2 = {e2.label}@{e2.timestamp}")

    test_logger(1)
    print("Testing event logger:")
    logger = EventLogger()
    for i, (l, ts) in enumerate(zip(logger.labels, logger.timestamps)):
        print(f"e{i+1} = {l}@{ts}")

    test_inplace_logger(1)
    print("Testing inplace event logger:")
    logger = EventLogger()
    for i, (l, ts) in enumerate(zip(logger.labels, logger.timestamps)):
        print(f"e{i+1} = {l}@{ts}")
