# generates a dictionary with the paths for the metrics file

import pandas as pd
import argparse
import os
import pathlib
import glob
from typing import Union
import npc_lims


parser = argparse.ArgumentParser()
parser.add_argument('--mouseID', help='Mouse ID of session', required=True)

# gets the directories for relevatn mouse
def get_metrics_directory(base_path: Union[str, pathlib.Path],  mouse_id: str):
    directories = os.listdir(base_path)
    probe_directories = []

    for d in directories:
        if mouse_id in d and 'json' not in d and 'Shortcut' not in d:
            probe_directories.append(d)
    
    return probe_directories

# gets the path for the metrics csv
def generate_metrics_path_days_codeocean(base_path: pathlib.Path, mouse_id: str) -> dict[int, str]:
    """
    >>> np_exp_path = pathlib.Path('//allen/programs/mindscope/workgroups/np-exp')
    >>> metrics_paths = generate_metrics_path_days_codeocean(np_exp_path, '681532')
    >>> metrics_paths[1][0]
    '//allen/programs/mindscope/workgroups/np-exp/681532_2023-10-16_0/probeA/continuous/Neuropix-PXI-100.0/metrics.csv'
    """
    sessions_mouse: list[npc_lims.SessionInfo] = []
    for session in npc_lims.get_session_info():
        if session.is_ephys and session.subject == mouse_id:
            sessions_mouse.append(session)

    dates = sorted([session.date for session in sessions_mouse])
    mouse_dirs = get_metrics_directory(base_path, mouse_id)
    days = [(dates.index(date) + 1, f'{mouse_id}_{date}_0') for date in dates if f'{mouse_id}_{date}_0' in mouse_dirs]
    metrics_path_days = {}

    for pair in days:
        session = pair[1]
        day = pair[0]
        metrics_files = list(base_path.glob(f'{session}/*/*/*/metrics.csv'))
        metrics_files = [metric_file.as_posix() for metric_file in metrics_files]
        metrics_path_days[day] = metrics_files
 
    return metrics_path_days

if __name__ == '__main__':
    import doctest

    doctest.testmod(
        optionflags=(doctest.IGNORE_EXCEPTION_DETAIL | doctest.NORMALIZE_WHITESPACE)
    )