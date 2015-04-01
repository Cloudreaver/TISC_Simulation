#!/usr/bin/env python

import ROOT
import numpy as np

print "Writing a tree"

f = ROOT.TFile("tree.root", "recreate")
t = ROOT.TTree("name_of_tree", "tree title")


# create 1 dimensional float arrays (python's float datatype corresponds to c++ doubles)
# as fill variables
n = np.zeros(1)
u = np.zeros(1)

# create the branches and assign the fill-variables to them
t.Branch('normal', n, 'normal/D')
t.Branch('uniform', u, 'uniform/D')

test_n = [1,3,2,3,4,1]
test_u = [2,2,3,1,3,2,1,3]

# create some random numbers, fill them into the fill varibles and call Fill()
for i in xrange(5):
	n[0] = test_n[i]#ROOT.gRandom.Gaus()
	u[0] = test_u[i]#ROOT.gRandom.Uniform()
	print n[0]
	print u[0]
	t.Fill()

# write the tree into the output file and close the file
f.Write()
f.Close()
