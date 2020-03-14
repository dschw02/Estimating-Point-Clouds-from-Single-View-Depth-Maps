import net_config_loader
import net_constructor
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('-p', nargs = '?', help = 'path to experiment')
args = parser.parse_args()

net_config_loader.create_net_configuration(args.p)