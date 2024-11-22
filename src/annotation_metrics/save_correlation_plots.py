import pathlib
import requests
import npc_lims
import npc_session
import warnings
import argparse
import numpy as np
import numpy.typing as npt
import pandas as pd
from get_correlation_plot import get_correlation_data
import npc_ephys
import re
import xmltodict
import upath
import io
import npc_sessions

parser = argparse.ArgumentParser()
parser.add_argument('--mouseID', help='Mouse ID of session')

SAMPLING_RATE = 30000.
BASE_PATH = pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/tissuecyte/plots')
NUM_CHANNELS = 384



def save_for_annotation(session: npc_session.SessionRecord, spike_times: np.ndarray,
                        spike_depths: np.ndarray, peak_channels: list[np.intp], probe: str, epoch:str) -> None:
    # maintain same structure as dynamic gating
    df_metrics = pd.DataFrame({'peak_channel': peak_channels})

    session_path = BASE_PATH / session.id
    if not session_path.exists():
        session_path.mkdir()
    
    probe_path = session_path / f'probe{probe}'
    if not probe_path.exists():
        probe_path.mkdir()

    if not (probe_path / 'continuous').exists():
        (probe_path / 'continuous').mkdir()
    
    if not (probe_path / 'continuous' / 'Neuropix-PXI-100.0').exists():
        (probe_path / 'continuous' / 'Neuropix-PXI-100.0').mkdir()
    
    session_metrics_path = probe_path / 'continuous' / 'Neuropix-PXI-100.0'
    df_metrics.to_csv(session_metrics_path / 'metrics.csv', index=False)
    np.save(session_metrics_path / f'spike_times_{epoch}.npy', spike_times)
    np.save(session_metrics_path / f'spike_depths_{epoch}.npy', spike_depths)

def get_channel_positions(settings_xml_path:upath.UPath) -> np.ndarray:
    with io.BytesIO(settings_xml_path.read_bytes()) as f:
        settings_dict = xmltodict.parse(f.read())
    
    electrode_xpositions = list(settings_dict['SETTINGS']['SIGNALCHAIN'][0]['PROCESSOR'][0]['EDITOR']['NP_PROBE'][0]['ELECTRODE_XPOS'].values())
    electrode_ypositions = list(settings_dict['SETTINGS']['SIGNALCHAIN'][0]['PROCESSOR'][0]['EDITOR']['NP_PROBE'][0]['ELECTRODE_YPOS'].values())

    electrode_xpositions = [int(electrode_xposition) for electrode_xposition in electrode_xpositions]
    electrode_ypositions = [int(electrode_yposition) for electrode_yposition in electrode_ypositions]

    channel_positions = [[electrode_xpositions[i], electrode_ypositions[i]] for i in range(len(electrode_xpositions))]
    return np.array(channel_positions)

def save_metrics_for_correlation(session: npc_session.SessionRecord, probe: npc_session.ProbeRecord,
                                 spike_times_aligned: npt.NDArray[np.float64], spike_clusters: npt.NDArray[np.int64],
                                 spike_interface_data: npc_ephys.SpikeInterfaceKS25Data, channel_positions: npt.NDArray[np.int64],
                                 peak_channels: list[np.intp], time_indices: npt.NDArray[np.intp] | None=None,
                                 tag='full') -> None:

    if tag != 'full':
        spike_times = spike_times_aligned[time_indices].T[0]

        """
        cluster_difference = np.max(spike_clusters) - len(spike_interface_data.unit_locations(probe)) + 1
        if cluster_difference > 0:
            spike_clusters = spike_clusters - cluster_difference
        """
        spike_clusters = spike_clusters[time_indices].T[0]
        assert len(spike_times) == len(spike_clusters)
        clusters_depths = channel_positions[peak_channels, 1]
        spike_depths = clusters_depths[spike_clusters]

        save_for_annotation(session, spike_times, spike_depths, peak_channels, probe, tag)
    else:
        """
        cluster_difference = np.max(spike_clusters) - len(spike_interface_data.unit_locations(probe)) + 1
        if cluster_difference > 0:
            spike_clusters = spike_clusters - cluster_difference
        """
        assert len(spike_times_aligned) == len(spike_clusters)
        clusters_depths = channel_positions[peak_channels, 1]
        spike_depths = clusters_depths[spike_clusters]
        save_for_annotation(session, spike_times_aligned, spike_depths, peak_channels, probe, tag)

def save_refinement_metrics(subject_id: str, session: npc_session.SessionRecord, session_surface: npc_session.SessionRecord | None=None) -> None:
    spike_interface_data = npc_ephys.SpikeInterfaceKS25Data(session)
    probes = spike_interface_data.probes

    #session_experiment = npc_sessions.DynamicRoutingSession(session)
    #epochs = session_experiment.epochs[:]

    for probe in probes:
        try:
            peak_channels = list(npc_ephys.get_amplitudes_waveforms_channels_ks25(spike_interface_data, electrode_group_name=probe).peak_channels)
            spike_times = spike_interface_data.spike_indexes(probe)
            spike_clusters = spike_interface_data.unit_indexes(probe)
            channel_positions = get_channel_positions(upath.UPath(r"\\allen\programs\mindscope\workgroups\np-behavior\tissuecyte\plots\settings_main.xml"))

            if session_surface is not None:
                spike_interface_data_surface = npc_ephys.SpikeInterfaceKS25Data(session_surface)
                peak_channels_surface = list(npc_ephys.get_amplitudes_waveforms_channels_ks25(spike_interface_data_surface, electrode_group_name=probe).peak_channels)
                spike_times_surface = spike_interface_data_surface.spike_indexes(probe)
                spike_clusters_surface = spike_interface_data_surface.unit_indexes(probe)
                channel_positions_surface = get_channel_positions(upath.UPath(r"\\allen\programs\mindscope\workgroups\np-behavior\tissuecyte\plots\settings_surface.xml"))

                peak_channels_surface = [peak_channel + 384 for peak_channel in peak_channels_surface]
                spike_times_surface = spike_times_surface + (spike_times[-1] + 1)
                spike_clusters_surface = spike_clusters_surface + (np.max(spike_clusters) + 1)

                channel_positions = np.concatenate((channel_positions, channel_positions_surface))
                peak_channels = np.concatenate((np.array(peak_channels), np.array(peak_channels_surface)))
                spike_clusters = np.concatenate((spike_clusters, spike_clusters_surface))
                spike_times = np.concatenate((spike_times, spike_times_surface))

            device_timing_on_sync = next(npc_ephys.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3(session), npc_lims.get_recording_dirs_experiment_path_from_s3(session), only_devices_including=probe))
            spike_times_aligned = npc_ephys.get_aligned_spike_times(spike_times, device_timing_on_sync)
  
            """
            for tag in epochs.tags:
                if 'task' in tag:
                    index = epochs.tags.tolist().index(tag)
                    task_times = epochs.iloc[index]
                    time_indices = np.argwhere((spike_times_aligned >= task_times['start_time']) & 
                                                        (spike_times_aligned <= task_times['stop_time']))
                    
                    save_metrics_for_correlation(session, probe, spike_times_aligned, spike_clusters, spike_interface_data, channel_positions,
                                                peak_channels, time_indices=time_indices, tag='task')
                    get_correlation_data(subject_id, tag='task')
                elif 'spontaneous' in tag:
                    index = epochs.tags.tolist().index(tag)
                    task_times = epochs.iloc[index]
                    time_indices = np.argwhere((spike_times_aligned >= task_times['start_time']) & 
                                                        (spike_times_aligned <= task_times['stop_time']))
                    
                    save_metrics_for_correlation(session, probe, spike_times_aligned, spike_clusters, spike_interface_data, channel_positions,
                                                peak_channels, time_indices=time_indices, tag='spont')
                    get_correlation_data(subject_id, tag='spont')
            """
            save_metrics_for_correlation(session, probe, spike_times_aligned, spike_clusters, spike_interface_data, channel_positions,
                                                peak_channels)
            get_correlation_data(subject_id, tag='full')
        except (ValueError, Exception, IndexError) as e:
            print(e)
            print(f'Failed to get metrics for session {session} and probe {probe}')
            pass
        

def get_annotation_data_for_mouse(mouse_id:str) -> None:
    sessions = npc_lims.get_sessions_with_data_assets(mouse_id)
    for session in sessions:
        session_surface = None
        try:
            if npc_lims.get_session_info(session).is_surface_channels:
                session_surface = npc_session.SessionRecord(session).with_idx(1)
            save_refinement_metrics(mouse_id, session, session_surface=session_surface)
        except (IndexError, ValueError, FileNotFoundError, npc_lims.exceptions.NoSessionInfo) as e:
            print(e)
            print(f'Failed to get metrics for session" {session}')
            pass
    
    
if __name__ == '__main__':
    #args = parser.parse_args()
    #mouse_id = args.mouseID
    #mouse_id = ['620263', '620264', '626791', '628801', '636397', '644547', '646318', '636766', '644864', '644866', '649943',
               #'644867', '649944', '662983', '668759', '670181', '670180', '670248', '660023', '666986', '668755', '667252',
                #'674562', '681532', '686740', '664851', '690706', '686176', '676909']
    mouse_id = ['741137']
    for mid in mouse_id:
        get_annotation_data_for_mouse(mid)