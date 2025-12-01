#!/bin/sh

# Test that we can invoke the spnl API without error
spnl -b email -m spnl/$MODEL -n 1 -l 1 --time gen1
