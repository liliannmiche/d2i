# -*- coding: utf-8 -*-
"""Starts the queue server and runs forever.
A port will be busy as long as the server is running.

Manual operation: just start the script, manually close when finished.

Script operation: start with p=subprocess.Popen(...), call p.poll() to wait for
                  an actual start, terminate with p.terminate() which takes
                  around 20 seconds to actually close the process.

Created on Sat Apr  6 21:01:26 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
import socket
from multiprocessing.managers import SyncManager
import Queue
import os


def CreateDoubleQueueServer(host, port, key, queue_size):
    
    qtask = Queue.Queue(maxsize=queue_size)
    qresult = Queue.Queue(maxsize=queue_size)
    
    class QueueManager(SyncManager):
        pass

    QueueManager.register('qtask', callable=lambda:qtask)
    QueueManager.register('qresult', callable=lambda:qresult)
    manager = QueueManager(address=(host, port), authkey=key)
    #manager.start()  # actually start the server
    manager.get_server().serve_forever()
    return manager


if __name__ == "__main__":

    host_file = open(cf._host, "w")
    host_ip = socket.gethostbyname(socket.gethostname())
    host_file.write(host_ip)
    host_file.close()

    mng = CreateDoubleQueueServer(host_ip, cf._port, cf._key, cf._qsize)
    print "Started queue server at %s" % host_ip












