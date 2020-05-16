#!/usr/bin/env python
# -*- coding: utf-8 -*-


from socket    import gethostname
from functools import wraps
from math      import floor
from time      import time, strftime, gmtime


from ..util import Singleton


class Event(object):


    def __init__(self, label="", accuracy=1000):

        self._label    = label
        self._accuracy = accuracy
        self._t_evt    = self._time(self._accuracy)


    @staticmethod
    def _time(accuracy):

        t = time()
        s = int(floor(t))
        return (s, int(round((t - s) * accuracy)))


    @property
    def accuracy(self):
        return self._accuracy


    @property
    def timestamp(self):

        t, s = self._t_evt
        return strftime("%Y-%m-%dT%H:%MZ%S", gmtime(t)) + (".%03d" % s)


    @property
    def label(self):
        return self._label



class EventLogger(object, metaclass=Singleton):

    def __init__(self):
        self.clear()
        self._hostname = gethostname()


    def add(self, evt):
        self._events.append(evt)


    def clear(self):
        self._events = list()


    @property
    def hostname(self):
        return self._hostname


    @property
    def events(self):
        return self._events


    @property
    def timestamps(self):
        for e in self.events:
            yield e.timestamp


    @property
    def labels(self):
        for e in self.events:
            yield e.label



#
# Functions to operate on the singleton EventLogger
#


def event_here(label, status=None):

    if status == None:
        status = "    "

    EventLogger().add(Event(label=status+","+label))



def start(label):
    event_here(label, status="beg ")



def stop(label):
    event_here(label, status="end ")



def event_log():
    hostname = EventLogger().hostname
    for e in EventLogger().events:
        yield f"{hostname},{e.timestamp},{e.label}"


#
# Decorator to log function calls
#
def log(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        event_here(func.__name__, status="call")
        ret = func(*args, **kwargs)
        event_here(func.__name__, status="ret ")
        return ret

    return wrapper
