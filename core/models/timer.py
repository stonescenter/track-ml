__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import datetime as dt
 
class Timer():
 
    def __init__(self):
        self.start_dt = None
 
    def start(self):
        self.start_dt = dt.datetime.now()
 
    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))