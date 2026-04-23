import xradar as xd
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import cmweather    # noqa 


def preprocess_radar_data(file_path, output_path, date=None, 
                          radar_field='corrected_reflectivity',
                          x_bounds=(-150000, 150000), y_bounds=(-150000, 150000),
                          **kwargs):
    """
    Preprocess cf/Radial radar data from a given file path. This module will load the radar data,
    then convert the 0.5 degree tilt to a .png image for model training.

    Parameters
    ----------
    file_path (str): Path to the radar data files.

    output_path (str): Path to save the processed .png images.
    date (str or list): Optional date string to filter radar files, in the format 'YYYYMMDD'.
    radar_field (str): The radar field to be processed, 
        default is 'corrected_reflectivity'.
    x_bounds (tuple): The x-axis bounds for plotting in meters.
    y_bounds (tuple): The y-axis bounds for plotting in meters.

    **kwargs:

    Additional Keyword Arguments are entered into the xarray plotting function.

    Returns
    -------
    label_df: pd.DataFrame
        DataFrame containing labels, paths, and times for the radar data.
    """
    
    file_list = glob.glob(file_path + '/*.nc')
    if date is not None:
        if isinstance(date, str):
            date = [date]
        for date_str in date:
            
            file_list = [f for f in file_list if date_str in f]
    out_df = pd.DataFrame(columns=['file_path', 'time', 'label', 'ref_min', 'ref_max'])
    if not "vmin" in kwargs:
        kwargs['vmin'] = -20
    if not "vmax" in kwargs:
        kwargs['vmax'] = 80
    if not "cmap" in kwargs:
        kwargs['cmap'] = 'ChaseSpectral'
    if "ax" in kwargs:
        return ValueError("Do not pass in an axis to this function.")
   
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in file_list:
        radar = xd.io.open_cfradial1_datatree(file)
        # Example preprocessing step: plot and save the 0.5 degree tilt
        radar = radar.xradar.georeference()
        if 'sweep_0' in radar:
            sweep = radar['sweep_0']
            if sweep["sweep_mode"] == 'ppi' or sweep["sweep_mode"] == 'sector':
                fig = plt.figure(figsize=(256/150, 256/150))
                ax = plt.axes()
                sweep["corrected_reflectivity"].where(
                        sweep["corrected_reflectivity"] > min_ref).plot(x="x", y="y",
                                                                        ax=ax,
                                                                        **kwargs)
                min_ref = sweep["corrected_reflectivity"].where(
                            sweep["corrected_reflectivity"] > min_ref).values.min()
                max_ref = sweep["corrected_reflectivity"].where(
                            sweep["corrected_reflectivity"] > min_ref).values.max()

                ax.set_xlim(x_bounds)
                ax.set_ylim(y_bounds)
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_xticks([-100000, -50000, 0, 50000, 100000])
                ax.set_yticks([-100000, -50000, 0, 50000, 100000])
                fig.tight_layout()
                file_name = os.path.join(output_path,
                                         os.path.basename(file).replace('.nc', '.png'))
                time_str = pd.to_datetime(sweep["time"].values[0]).strftime('%Y-%m-%d %H:%M:%S')
                label = "UNKNOWN"   # Placeholder for actual label extraction logic
                fig.savefig(os.path.join(output_path,
                                         os.path.basename(file).replace('.nc', '.png')),
                            dpi=150)
                plt.close(fig)
                out_df.loc[len(out_df)] = [file_name, time_str, label, min_ref, max_ref]

            else:
                print(f"Sweep mode is not PPI or sector scan in {file}, skipping.")
        else:
            print(f"No sweep_0 found in {file}, skipping.")
    
    out_df.set_index('time', inplace=True)
    out_df = out_df.sort_index()
    return out_df



