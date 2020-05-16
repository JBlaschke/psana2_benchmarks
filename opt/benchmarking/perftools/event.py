#!/usr/bin/env python
# -*- coding: utf-8 -*-


from math import floor
from time import time, strftime, gmtime


from ..util import Singleton


class Event(object):


    def __init__(self, accuracy=1000):

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



class EventLogger(object, metaclass=Singleton):
    
    def __init__(self):
        self._events = list()


    def add(self, evt):
        self._events.append(evt)


    @property
    def events(self):
        return self._events


    @property
    def timestamps(self):
        for e in self.events:
            yield e.timestamp
