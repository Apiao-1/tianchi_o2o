import datetime


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

def init_logger(start=datetime.datetime.now()):
    global log
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    return log

def get_logger():
    global log
    return log

def set_logger(logger):
    global log
    log = logger

