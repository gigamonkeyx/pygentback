#!/bin/bash
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning"
python "$@"
