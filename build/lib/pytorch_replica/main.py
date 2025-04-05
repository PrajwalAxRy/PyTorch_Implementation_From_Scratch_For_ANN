import argparse
import sys
from . import __version__ 

def main():
    parser = argparse.ArgumentParser(prog='pytorch_replica')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args()