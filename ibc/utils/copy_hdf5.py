#from ../modules/hdf5_creator import create_empty_hdf5
from tables import openFile
import numpy as np


def copy_hdf5(data, new, batch=100000):
    """Copies websites from one database to another.
    """
    print "Copying websites"
    db0 = openFile(data, "r")
    db1 = openFile(new, "a")
    i = 0
    Ws0 = db0.root.Websites
    Img0 = db0.root.Images
    Reg0 = db0.root.Regions
    Des0 = db0.root.Descriptors
    Ws1 = db1.root.Websites
    Img1 = db1.root.Images
    Reg1 = db1.root.Regions
    Des1 = db1.root.Descriptors

    # websites
    N = Ws0.nrows
    for b in range(N/batch + 1):
        nmin = b*batch
        nmax = min((b+1)*batch, N)
        rows = []
        # just copy rows as they are the same
        Ws1.append(Ws0.read(nmin, nmax))
        print  "ws: %d/%d" % (nmax, N)
    Ws1.attrs.last_index = Ws0.attrs.last_index
    Ws1.flush()

    # images
    N = Img0.nrows    
    img_repr = np.ones((24,), dtype=np.float64) * -1
    for b in range(N/batch + 1):        
        nmin = b*batch
        nmax = min((b+1)*batch, N)
        rows = []
        for row in Img0.read(nmin, nmax):
            rows.append(tuple(row) + (img_repr,))
        Img1.append(rows)
        print  "img: %d/%d" % (nmax, N)
    Img1.attrs.last_index = Img0.attrs.last_index
    Img1.attrs.nr_in_class = Img0.attrs.nr_in_class
    Img1.flush()

    # regions
    N = Reg0.nrows    
    ngb = np.ones((10,2), dtype=np.float64) * -1
    for b in range(N/batch + 1):        
        nmin = b*batch
        nmax = min((b+1)*batch, N)
        rows = []
        for tupl in Reg0.read(nmin, nmax):
            row = list(tupl)
            # format rows here
            rows.append(tuple(row[:6] + [ngb] + row[6:]))
        Reg1.append(rows)
        print  "reg: %d/%d" % (nmax, N)
    Reg1.attrs.last_index = Reg0.attrs.last_index
    Reg1.flush()

    # descriptors
    N = Des0.nrows   
    for b in range(N/batch + 1):        
        nmin = b*batch
        nmax = min((b+1)*batch, N)
        Des1.append(Des0.read(nmin, nmax))
        print  "des: %d/%d" % (nmax, N)
    Des1.attrs.last_index = Des0.attrs.last_index
    Des1.flush()

    db0.close()
    db1.close()
    print 'Done copying!'
        
        
if __name__ == "__main__":
    copy_hdf5("/users/akusoka1/local/SPIIRAS/spiiras.h5", 
              "/data/spiiras.h5")



















































