#!/bin/bash

cmd_line="$@"

source venv/bin/activate
$cmd_line
deactivate

