import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import npc_lims, npc_ephys, npc_session
from sklearn.metrics import roc_curve, roc_auc_score
import spike_utils
from npc_sessions import DynamicRoutingSession
import os
import pickle

#calculate metrics for channel alignment

def compute_metrics_for_alignment(trials, units, session_info, save_path):

    stim_context_modulation = {
        'unit_id':[],
        'session_id':[],
        'experiment_day':[],
        'project':[],
        'probe':[],
        'peak_channel':[],
        'lick_modulation_roc_auc':[],
        'vis_discrim_roc_auc':[],
        'aud_discrim_roc_auc':[],
        'any_vis_roc_auc':[],
        'any_aud_roc_auc':[],
        'firing_rate':[],
        'peak_to_valley':[],
        'peak_trough_ratio':[],
        'repolarization_slope':[],
        'recovery_slope':[],
        'spread':[],
        'velocity_above':[],
        'velocity_below':[],
        'snr':[],

        'amplitude_cutoff':[], 
        'amplitude_cv_median':[], 
        'amplitude_cv_range':[],
        'amplitude_median':[], 
        'drift_ptp':[], 
        'drift_std':[], 
        'drift_mad':[],
        'firing_range':[], 
        'isi_violations_ratio':[],
        'isi_violations_count':[], 
        'presence_ratio':[],
        'rp_contamination':[], 
        'rp_violations':[], 
        'sliding_rp_violation':[],
        'sync_spike_2':[], 
        'sync_spike_4':[], 
        'sync_spike_8':[], 
        'd_prime':[],
        'isolation_distance':[], 
        'l_ratio':[], 
        'silhouette':[], 
        'nn_hit_rate':[],
        'nn_miss_rate':[], 
        'exp_decay':[], 
        'half_width':[], 
        'num_negative_peaks':[],
        'num_positive_peaks':[],
    }

    if trials is not None:
        contexts=trials['context_name'].unique()

        if 'Templeton' in session_info.project:
            contexts = ['aud','vis']

            start_time=trials['start_time'].iloc[0]
            fake_context=np.full(len(trials), fill_value='nan')
            fake_block_nums=np.full(len(trials), fill_value=np.nan)

            if np.random.choice(contexts,1)=='vis':
                block_contexts=['vis','aud','vis','aud','vis','aud']
            else:
                block_contexts=['aud','vis','aud','vis','aud','vis']

            trials['true_block_index']=trials['block_index']
            trials['true_context_name']=trials['context_name']

            for block in range(0,6):
                block_start_time=start_time+block*10*60
                block_end_time=start_time+(block+1)*10*60
                block_trials=trials.query('start_time>=@block_start_time').index
                fake_context[block_trials]=block_contexts[block]
                fake_block_nums[block_trials]=block
            
            trials['context_name']=fake_context
            trials['block_index']=fake_block_nums
            trials['is_vis_context']=trials['context_name']=='vis'
            trials['is_aud_context']=trials['context_name']=='aud'

        #make data array first
        time_before = 0.5
        time_after = 0.5
        binsize = 0.025
        trial_da = spike_utils.make_neuron_time_trials_tensor(units, trials, time_before, time_after, binsize)

    #for each unit
    for uu,unit in units.iterrows():
        
        stim_context_modulation['unit_id'].append(unit['unit_id'])
        stim_context_modulation['session_id'].append(str(session_info.id))
        stim_context_modulation['project'].append(str(session_info.project))
        stim_context_modulation['experiment_day'].append(str(session_info.experiment_day))
        stim_context_modulation['probe'].append(str(unit['electrode_group_name']))
        stim_context_modulation['peak_channel'].append(unit['peak_channel'])

        stim_context_modulation['firing_rate'].append(unit['firing_rate'])
        stim_context_modulation['peak_to_valley'].append(unit['peak_to_valley'])
        stim_context_modulation['peak_trough_ratio'].append(unit['peak_trough_ratio'])
        stim_context_modulation['repolarization_slope'].append(unit['repolarization_slope'])
        stim_context_modulation['recovery_slope'].append(unit['recovery_slope'])

        stim_context_modulation['spread'].append(unit['spread'])
        stim_context_modulation['velocity_above'].append(unit['velocity_above'])
        stim_context_modulation['velocity_below'].append(unit['velocity_below'])
        stim_context_modulation['snr'].append(unit['spread'])

        stim_context_modulation['amplitude_cutoff'].append(unit['amplitude_cutoff'])
        stim_context_modulation['amplitude_cv_median'].append(unit['amplitude_cv_median'])
        stim_context_modulation['amplitude_cv_range'].append(unit['amplitude_cv_range'])
        stim_context_modulation['amplitude_median'].append(unit['amplitude_median'])
        stim_context_modulation['drift_ptp'].append(unit['drift_ptp'])
        stim_context_modulation['drift_std'].append(unit['drift_std'])
        stim_context_modulation['drift_mad'].append(unit['drift_mad'])
        stim_context_modulation['firing_range'].append(unit['firing_range'])
        stim_context_modulation['isi_violations_ratio'].append(unit['isi_violations_ratio'])
        stim_context_modulation['isi_violations_count'].append(unit['isi_violations_count'])
        stim_context_modulation['presence_ratio'].append(unit['presence_ratio'])
        stim_context_modulation['rp_contamination'].append(unit['rp_contamination'])
        stim_context_modulation['rp_violations'].append(unit['rp_violations'])
        stim_context_modulation['sliding_rp_violation'].append(unit['sliding_rp_violation'])
        stim_context_modulation['sync_spike_2'].append(unit['sync_spike_2'])
        stim_context_modulation['sync_spike_4'].append(unit['sync_spike_4'])
        stim_context_modulation['sync_spike_8'].append(unit['sync_spike_8'])
        stim_context_modulation['d_prime'].append(unit['d_prime'])
        stim_context_modulation['isolation_distance'].append(unit['isolation_distance'])
        stim_context_modulation['l_ratio'].append(unit['l_ratio'])
        stim_context_modulation['silhouette'].append(unit['silhouette'])
        stim_context_modulation['nn_hit_rate'].append(unit['nn_hit_rate'])
        stim_context_modulation['nn_miss_rate'].append(unit['nn_miss_rate'])
        stim_context_modulation['exp_decay'].append(unit['exp_decay'])
        stim_context_modulation['half_width'].append(unit['half_width'])
        stim_context_modulation['num_negative_peaks'].append(unit['num_negative_peaks'])
        stim_context_modulation['num_positive_peaks'].append(unit['num_positive_peaks'])
        
        #if surface channel, don't try to calculate metrics
        if unit['peak_channel']>383:
            #append nans for all metrics
            stim_context_modulation['any_vis_roc_auc'].append(np.nan)
            stim_context_modulation['any_aud_roc_auc'].append(np.nan)
            stim_context_modulation['vis_discrim_roc_auc'].append(np.nan)
            stim_context_modulation['aud_discrim_roc_auc'].append(np.nan)
            stim_context_modulation['lick_modulation_roc_auc'].append(np.nan)
            continue
        
        if trials is not None:
            #find baseline frs across all trials
            baseline_frs = trial_da.sel(unit_id=unit['unit_id'],time=slice(-0.1,0)).mean(dim='time')

            all_stim_frs_by_trial = {}
            #loop through stimuli
            for ss in trials['stim_name'].unique():

                #stimulus modulation
                if "Templeton" in session_info.project:
                    stim_trials = trials.query('stim_name==@ss')
                else:
                    stim_trials = trials.query('stim_name==@ss and is_response==False') #remove response trials to minimize contamination
                stim_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=stim_trials.index).mean(dim='time',skipna=True)

                all_stim_frs_by_trial[ss]=stim_frs_by_trial

            if "Templeton" in session_info.project:
                any_vis_trials = trials.query('stim_name.str.contains("vis")')
            else:
                any_vis_trials = trials.query('stim_name.str.contains("vis") and is_response==False')
            any_vis_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=any_vis_trials.index).mean(dim='time',skipna=True)
            any_vis_baseline_frs_by_trial = baseline_frs.sel(trials=any_vis_trials.index)
            any_vis_and_baseline_frs=np.concatenate([any_vis_frs_by_trial.values,any_vis_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(any_vis_frs_by_trial)),np.zeros(len(any_vis_baseline_frs_by_trial))])
            if len(np.unique(binary_label))>1:
                any_vis_context_auc=roc_auc_score(binary_label,any_vis_and_baseline_frs)
            else:
                any_vis_context_auc=np.nan
            stim_context_modulation['any_vis_roc_auc'].append(any_vis_context_auc)

            if "Templeton" in session_info.project:
                any_aud_trials = trials.query('stim_name.str.contains("sound")')
            else:
                any_aud_trials = trials.query('stim_name.str.contains("sound") and is_response==False')
            any_aud_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0,0.1),trials=any_aud_trials.index).mean(dim='time',skipna=True)
            any_aud_baseline_frs_by_trial = baseline_frs.sel(trials=any_aud_trials.index)
            any_aud_and_baseline_frs=np.concatenate([any_aud_frs_by_trial.values,any_aud_baseline_frs_by_trial.values])
            binary_label=np.concatenate([np.ones(len(any_aud_frs_by_trial)),np.zeros(len(any_aud_baseline_frs_by_trial))])
            if len(np.unique(binary_label))>1:
                any_aud_context_auc=roc_auc_score(binary_label,any_aud_and_baseline_frs)
            else:
                any_aud_context_auc=np.nan
            stim_context_modulation['any_aud_roc_auc'].append(any_aud_context_auc)

            #same modality stimulus discrimination
            #vis1 vs. vis2
            vis1_and_vis2_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['vis2'].values])
            binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])),np.zeros(len(all_stim_frs_by_trial['vis2']))])
            if len(np.unique(binary_label))>1:
                vis_discrim_auc=roc_auc_score(binary_label,vis1_and_vis2_frs)
            else:
                vis_discrim_auc=np.nan
            stim_context_modulation['vis_discrim_roc_auc'].append(vis_discrim_auc)

            #aud1 vs. aud2
            aud1_and_aud2_frs=np.concatenate([all_stim_frs_by_trial['sound1'].values,all_stim_frs_by_trial['sound2'].values])
            binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['sound1'])),np.zeros(len(all_stim_frs_by_trial['sound2']))])
            if len(np.unique(binary_label))>1:
                aud_discrim_auc=roc_auc_score(binary_label,aud1_and_aud2_frs)
            else:
                aud_discrim_auc=np.nan
            stim_context_modulation['aud_discrim_roc_auc'].append(aud_discrim_auc)

            # #targets: vis1 vs sound1
            # vis1_vs_aud1_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['sound1'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])),np.zeros(len(all_stim_frs_by_trial['sound1']))])
            # target_discrim_auc=roc_auc_score(binary_label,vis1_vs_aud1_frs)
            # stim_context_modulation['target_discrim_roc_auc'].append(target_discrim_auc)

            # #nontargets: vis2 vs sound2
            # vis2_vs_aud2_frs=np.concatenate([all_stim_frs_by_trial['vis2'].values,all_stim_frs_by_trial['sound2'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis2'])),np.zeros(len(all_stim_frs_by_trial['sound2']))])
            # nontarget_discrim_auc=roc_auc_score(binary_label,vis2_vs_aud2_frs)
            # stim_context_modulation['nontarget_discrim_roc_auc'].append(nontarget_discrim_auc)

            # #vis vs. aud
            # vis_and_aud_frs=np.concatenate([all_stim_frs_by_trial['vis1'].values,all_stim_frs_by_trial['vis2'].values,
            #                                 all_stim_frs_by_trial['sound1'].values,all_stim_frs_by_trial['sound2'].values])
            # binary_label=np.concatenate([np.ones(len(all_stim_frs_by_trial['vis1'])+len(all_stim_frs_by_trial['vis2'])),
            #                             np.zeros(len(all_stim_frs_by_trial['sound1'])+len(all_stim_frs_by_trial['sound2']))])
            # vis_vs_aud_auc=roc_auc_score(binary_label,vis_and_aud_frs)
            # stim_context_modulation['vis_vs_aud_roc_auc'].append(vis_vs_aud_auc)

            #lick modulation
            if "DynamicRouting" in session_info.project:
                lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==True) or \
                                        (stim_name=="sound1" and context_name=="vis" and is_response==True)')
                non_lick_trials = trials.query('(stim_name=="vis1" and context_name=="aud" and is_response==False) or \
                                                (stim_name=="sound1" and context_name=="vis" and is_response==False)')
            elif "Templeton" in session_info.project:
                lick_trials = trials.query('is_response==True')
                non_lick_trials = trials.query('is_response==False')

            lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=lick_trials.index).mean(dim='time',skipna=True)
            non_lick_frs_by_trial = trial_da.sel(unit_id=unit['unit_id'],time=slice(0.2,0.5),trials=non_lick_trials.index).mean(dim='time',skipna=True)

            #ROC AUC
            binary_label = np.concatenate([np.ones(lick_frs_by_trial.size),np.zeros(non_lick_frs_by_trial.size)])
            binary_score = np.concatenate([lick_frs_by_trial.values,non_lick_frs_by_trial.values])
            if len(np.unique(binary_label))>1:
                lick_roc_auc = roc_auc_score(binary_label, binary_score)
            else:
                lick_roc_auc = np.nan
            stim_context_modulation['lick_modulation_roc_auc'].append(lick_roc_auc)
        
        else:
            stim_context_modulation['any_vis_roc_auc'].append(np.nan)
            stim_context_modulation['any_aud_roc_auc'].append(np.nan)
            stim_context_modulation['vis_discrim_roc_auc'].append(np.nan)
            stim_context_modulation['aud_discrim_roc_auc'].append(np.nan)
            stim_context_modulation['lick_modulation_roc_auc'].append(np.nan)
    
    stim_context_modulation = pd.DataFrame(stim_context_modulation)
    stim_context_modulation['visual_response'] = np.abs(stim_context_modulation['any_vis_roc_auc'] - 0.5)*2
    stim_context_modulation['auditory_response'] = np.abs(stim_context_modulation['any_aud_roc_auc'] - 0.5)*2
    stim_context_modulation['visual_discrim'] = np.abs(stim_context_modulation['vis_discrim_roc_auc'] - 0.5)*2
    stim_context_modulation['auditory_discrim'] = np.abs(stim_context_modulation['aud_discrim_roc_auc'] - 0.5)*2
    stim_context_modulation['lick_modulation'] = np.abs(stim_context_modulation['lick_modulation_roc_auc'] - 0.5)*2


    stim_context_modulation.drop(columns=['any_vis_roc_auc','any_aud_roc_auc','vis_discrim_roc_auc','aud_discrim_roc_auc','lick_modulation_roc_auc'],inplace=True)

    probes=stim_context_modulation['probe'].unique()
    for probe in probes:
        probe_units=stim_context_modulation.query('probe==@probe')
        probe_units.to_csv(os.path.join(save_path,session_info.id+'_day_'+str(session_info.experiment_day)+'_'+probe+'_stim_modulation.csv'),index=False)

def save_stim_metrics_for_alignment(session_id: str):
    #get units from npc_ephys
    save_path=r"\\allen\programs\mindscope\workgroups\np-behavior\tissuecyte\metrics for alignment\stimulus responsiveness"
    save_path_wf=r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\metrics for alignment\peak waveforms"
    #fix to try probe-by-probe

    save_waveforms=True
    compute_waveform_PCs=True
    overwrite=False

    except_dict={}

    session_info = npc_lims.get_session_info(session_id)

    try:
        print(session_info.id)
        si = npc_ephys.SpikeInterfaceKS25Data(session_info.id)
        #check for surface channel asset
        if session_info.is_surface_channels:
            surface_channel_session_id=npc_session.SessionRecord(session_info.id).with_idx(1)
            #only use surface channel recordings sorted recently
            surface_flag=True
            si_surface_channels=npc_ephys.SpikeInterfaceKS25Data(surface_channel_session_id)
        else:
            surface_flag=False
            si_surface_channels=None
        try:
            trials=pd.read_parquet(
                                npc_lims.get_cache_path('trials',session_info.id,version='any')
                            )
        except:
            print('no cached trials table, using npc_sessions')
            session = DynamicRoutingSession(session_info.id)
            trials = session.trials[:]
            #  trials=None
        for probe in si.probes:
            #check for existing file(s)
            
            try:
                if ((os.path.exists(os.path.join(save_path,session_info.id+'_day_'+str(session_info.experiment_day)+'_probe'+probe+'_stim_modulation.csv')) and 
                    (os.path.exists(os.path.join(save_path_wf,session_info.id+'_day_'+str(session_info.experiment_day)+'_'+probe+'_peak_waveforms.pkl')) and save_waveforms) 
                    and not overwrite and not surface_flag)):
                        continue
                device_timing_on_sync = npc_ephys.get_ephys_timing_on_sync(npc_lims.get_h5_sync_from_s3(session_info.id), 
                                                                            npc_lims.get_recording_dirs_experiment_path_from_s3(session_info.id),
                                                                            only_devices_including='Probe'+probe)
                
                units=npc_ephys.make_units_table_from_spike_interface_ks25(session_info.id, device_timing_on_sync)
                units=npc_ephys.add_global_unit_ids(units,session_info.id)

                if surface_flag:
                    units_surface=si_surface_channels.quality_metrics_df(probe=probe)
                    units_surface=pd.concat([units_surface,si_surface_channels.template_metrics_df(probe=probe)],axis=1)
                    units_surface['peak_channel']=npc_ephys.get_amplitudes_waveforms_channels_ks25(si_surface_channels, electrode_group_name=probe).peak_channels

                    if units_surface['peak_channel'].max()<384:
                        units_surface['peak_channel']+=384
                    units_surface['electrode_group_name']='probe'+probe
                    units_surface['cluster_id']=np.arange(0,len(units_surface))
                    # add global unit IDs}
                    units_surface=npc_ephys.add_global_unit_ids(units_surface,surface_channel_session_id)
                    #append to session units table
                    units=pd.concat([units,units_surface],axis=0,ignore_index=True)


                #add surface channel flag - or automatically deal with nans or no spikes
                compute_metrics_for_alignment(trials, units, session_info, save_path)

                #get waveforms for units
                if save_waveforms:
                    peak_waveforms={
                        'unit_id':[],
                        'session_id':[],
                        'peak_channel':[],
                        'probe':[],
                        'waveform':[],
                    }
                    waveforms=si.get_nwb_units_device_property('waveform_mean','probe'+probe)
                    if surface_flag:
                        #get surface channel waveforms
                        waveforms_surface=si_surface_channels.get_nwb_units_device_property('waveform_mean','probe'+probe)
                        #append to session waveforms
                        waveforms=np.concatenate([waveforms,waveforms_surface],axis=0)
                            
                    for uu,(_,unit) in enumerate(units.iterrows()):
                        peak_waveforms['unit_id'].append(unit['unit_id'])
                        peak_waveforms['session_id'].append(str(session_info.id))
                        peak_waveforms['peak_channel'].append(unit['peak_channel'])
                        peak_waveforms['probe'].append(probe)
                        if unit['peak_channel']>383: 
                            #multi-channel waveform has 384, not 768 channels - subtract 384 from surface channels for correct index
                            unit_peak_channel=unit['peak_channel']-384
                        else:
                            unit_peak_channel=unit['peak_channel']
                        peak_waveforms['waveform'].append(waveforms[uu,:,unit_peak_channel])
                    peak_waveforms=pd.DataFrame(peak_waveforms)
                    peak_waveforms.to_pickle(os.path.join(save_path_wf,session_info.id+'_day_'+str(session_info.experiment_day)+'_'+probe+'_peak_waveforms.pkl'))

                    if compute_waveform_PCs:
                        pca_path=r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\metrics for alignment\waveform_pca_model.pkl"
                        pca_scaled_path=r"\\allen\programs\mindscope\workgroups\dynamicrouting\Ethan\metrics for alignment\waveform_pca_scaled_model.pkl"

                        waveforms_array=peak_waveforms['waveform'].values
                        waveforms_array=np.vstack(waveforms_array)
                        waveforms_array_scaled=waveforms_array/np.max(np.abs(waveforms_array),axis=1)[:,None]

                        pca=pickle.load(open(pca_path,'rb'))
                        pca_scaled=pickle.load(open(pca_scaled_path,'rb'))

                        waveforms_transformed=pca.transform(waveforms_array)
                        waveforms_scaled_transformed=pca_scaled.transform(waveforms_array_scaled)

                        waveform_pcs={
                            'unit_id':peak_waveforms['unit_id'].values,
                            'session_id':peak_waveforms['session_id'].values,
                            'peak_channel':peak_waveforms['peak_channel'].values,
                        }

                        #add PCs to dataframe
                        for i in range(3):
                            waveform_pcs['wf_PC'+str(i+1)]=waveforms_transformed[:,i]
                        for i in range(3):
                            waveform_pcs['wf_PC'+str(i+1)+'_scaled']=waveforms_scaled_transformed[:,i]

                        waveform_pcs=pd.DataFrame(waveform_pcs)

                        #hack to add probe as a columns
                        probe_list=[]
                        for xx in range(len(waveform_pcs)):
                            probe_list.append('probe'+waveform_pcs['unit_id'].iloc[xx].split('_')[-1][0])
                        waveform_pcs['probe']=probe_list

                        #load metrics, merge, and re-save
                        metrics=pd.read_csv(os.path.join(save_path,session_info.id+'_day_'+str(session_info.experiment_day)+'_probe'+probe+'_stim_modulation.csv'))
                        metrics=metrics.merge(waveform_pcs,on=['unit_id','session_id','peak_channel','probe'])
                        metrics.to_csv(os.path.join(save_path,session_info.id+'_day_'+str(session_info.experiment_day)+'_probe'+probe+'_stim_modulation.csv'),index=False)


            except Exception as e:
                print(e)
                except_dict[session_info.id+'_'+probe]=e
    except Exception as e:
        print(e)
        except_dict[session_info.id]=e

    #save except dict as pickle file
    with open(os.path.join(save_path,'except_dict.pkl'), 'wb') as f:
        pickle.dump(except_dict, f)

if __name__ == '__main__':
    session_id = '741137_2024-10-09'
    save_stim_metrics_for_alignment(session_id)