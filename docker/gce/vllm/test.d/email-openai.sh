#!/bin/sh

# Test that we can invoke the OpenAI API without error
spnl -b email -m openai/$MODEL -n 1 -l 1 --time gen1
