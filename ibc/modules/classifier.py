# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:30:16 2013

@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from tables import openFile
from elm.elm import ELM
import cPickle
import numpy as np



def get_data():
    """Automatically select between HDF5 and img_data.pkl.
    """
    if cf._mode == "hdf5":
        db = openFile(cf._hdf5, "r")
        Images = db.root.Images
        Websites = db.root.Websites
        n = Images.nrows
        X = np.empty((n, 2*cf._maxc))
        Y = np.zeros((n, cf._maxc)) - 1
        I = []
        i = 0
        # getting translation table for ws_index: url
        nw = Websites.nrows
        urls = {}
        for item in Websites.read(0, nw):
            urls[item[0]] = item[7]
        # gathering data
        for row in Images.iterrows():
            X[i,:] = row["img_repr"]
            Y[i, row["classN"]] = 1
            I.append(urls[row["site_index"]])
            i += 1
        db.close()
        return X,Y,I
    else:
        # assume that true class is unknown in demo mode
        D = cPickle.load(open(cf._img_data, "rb"))
        X = np.array([d[1] for d in D])
        WS = [d[0] for d in D]
        return X,[],WS
                

def train_elm():
    """Assume only trained on HDF5 file.
    """
    if cf._mode != "hdf5":
        print "Can train ELM only in batch mode."
        return
    X,Y,_ = get_data()
    # randomly shuffle the data
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    Y = Y[idx, :]
    Xt = np.empty((0, X.shape[1]))
    Yt = np.empty((0, Y.shape[1]))
    Xv = np.empty((0, X.shape[1]))
    Yv = np.empty((0, Y.shape[1]))
    for c in xrange(Y.shape[1]):  # for each class:
        idxc = np.where(Y[:,c] == 1)[0]
        if idxc != []:  # if there are any samples of that class:
            # calculate training and validation sample indexes
            l = len(idxc)
            j1 = min(l, cf._train_size)
            j2 = j1 + max(0, min(l - j1, cf._val_size))
            # gathering training and validation sets
            Xt = np.vstack((Xt, X[idxc[:j1], :]))
            Yt = np.vstack((Yt, Y[idxc[:j1], :]))
            Xv = np.vstack((Xv, X[idxc[j1:j2], :]))
            Yv = np.vstack((Yv, Y[idxc[j1:j2], :]))
            
    # normalizing data
    xm = np.mean(Xt, 0)
    xs = np.std(Xt, 0)
    xs[xs==0] = 1  # prevent dividing by zero for empty values
    Xt = (Xt - np.tile(xm, (Xt.shape[0], 1))) / np.tile(xs, (Xt.shape[0], 1))
    Xv = (Xv - np.tile(xm, (Xv.shape[0], 1))) / np.tile(xs, (Xv.shape[0], 1))
            
    # training the best elm
    elm = ELM()
    e_best = float("+inf")
    param = []
    for i in xrange(cf._elm_rep):
        elm.train_basic(Xt, Yt, {'lin':1, 'tanh':cf._neurons})
        _, e = elm.run(Xv, Yv)
        if e < e_best:
            e_best = e
            print "ebest = ", e
            param = elm.get_param()
            param["xm"] = xm
            param["xs"] = xs

    cPickle.dump(param, open(cf._elm_param, "wb"), -1)
    

def save_res(Yh, I):
    """Save results to HDF5 or formatted text document.
    """
    # determining results for websites
    wslist = list(set(I))
    res = np.zeros((len(wslist), Yh.shape[1]))
    for i in xrange(len(Yh)):
        idx = wslist.index(I[i])
        res[idx,:] += Yh[i,:]        
        
    f = open(cf._f_out, "w")
    for i in xrange(len(wslist)):
        line = "%s;%d;%s\n" % (wslist[i], np.argmax(res[i,:]), str(res[i,:]))
        f.write(line)
    f.close()


def run_elm(save_txt):
    X,Y,I = get_data()
    # initialize elm
    elm = ELM()
    param = cPickle.load(open(cf._elm_param, "rb"))
    elm.set_param(param)

    # normalize data
    xm = param["xm"]
    xs = param["xs"]
    X = (X - np.tile(xm, (X.shape[0], 1))) / np.tile(xs, (X.shape[0], 1))

    # run elm
    Yh = elm.run(X)
    print np.argmax(Yh, 1)
    # if true classes are known, show error
    if Y != []:
        err = np.sum(np.argmax(Yh, 1) == np.argmax(Y,1))
        err = float(err) / Y.shape[0]
        print np.argmax(Y,1)
        print "classification accuracy = %.03f" % err

    # save results
    if save_txt:
        save_res(Yh, I)    
    else:
        cPickle.dump((Yh,I), open(cf._f_out+".pkl", "wb"), -1)





















