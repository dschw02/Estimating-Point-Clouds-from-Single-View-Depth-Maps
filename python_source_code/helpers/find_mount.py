from os.path import join
from os.path import realpath

repo_name = "masters_thesis"
repository_path = join("/", *realpath(__file__).split("/")[0:realpath(__file__).split("/").index(repo_name) + 1])

