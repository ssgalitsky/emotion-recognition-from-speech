#! /usr/bin/env python
from voice import app
import os

port = int(os.getenv('PORT', 5000))
app.run(port=port, debug=True)
