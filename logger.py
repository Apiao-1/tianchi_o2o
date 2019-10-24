import datetime
import os

# class Logger(object):
#     def __init__(self, start=datetime.datetime.now()):
#         # self.server = redis.Redis(host='localhost', port='6379', db=0)
#         self.logger = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
#
#     def get_logger(self):
#         return self.logger
#
#     def set_logger(self, logger):
#         self.logger = logger
#

log = ''
start=datetime.datetime.now()

def init_logger():
    global log
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    return log

def get_logger():
    global log
    return log

def set_logger(logger):
    global log
    log = logger

def save(logger):
    global log
    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)

