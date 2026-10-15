#!/bin/bash
export FLASK_ENV=development # If your device's memory is less than or equal to 16GB, please comment out this line
python -m web_agent_site.app --log --attrs
