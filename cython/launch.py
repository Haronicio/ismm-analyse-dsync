import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*ChainedAssignmentError.*")
warnings.filterwarnings("ignore")
from adsync import run_main,run_build,run_personnal,run_interpersonnal
if __name__ == '__main__':
    run_interpersonnal()