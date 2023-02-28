from os import path
import git

def get_git_root(filepath: str = None):
    # Use the utils.py file as the default path
    filepath = path.abspath(__file__) if filepath is None else filepath
    
    # Traverse the repo until we find the .git folder
    git_repo = git.Repo(filepath, search_parent_directories=True)
    
    # Find the directory name of the git repo
    return git_repo.git.rev_parse("--show-toplevel") 

ROOT = get_git_root()

DIRS = {
    "DATA": path.join(ROOT, "data"),
    "MODELS": path.join(ROOT, "models"),
    "LATEX": path.join(ROOT, "latex"),
    "FIGURES": path.join(ROOT, "figures"),
}

FILES = {
    "TCR": path.join(DIRS["DATA"], "tcr.csv")
}