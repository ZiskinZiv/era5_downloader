#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:01:34 2021

@author: shlomi
ALGO for filling in missing data at 4Xdaily 1.25X1.25 deg ERA5 dataset:
    1) download the missing dates using max res., use check hourly time to determine
    which dates are missing
    2) run save_big_ERA5_dataset_as_yearly_files
    3)run concat_missing_daily_data_in_yearly_files
    4) run reconcat_all_yearly_files
    all should take little time and be done on the remote on ipython
"""

def path_glob(path, glob_str='*.nc', return_empty_list=False):
    """returns all the files with full path(pathlib3 objs) if files exist in
    path, if not, returns FilenotFoundErro"""
    from pathlib import Path
#    if not isinstance(path, Path):
#        raise Exception('{} must be a pathlib object'.format(path))
    path = Path(path)
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise FileNotFoundError('{} search in {} found no files.'.format(glob_str,
                                path))
    elif not files_with_path and return_empty_list:
        return files_with_path
    else:
        return files_with_path



def get_unique_index(da, dim='time', verbose=False):
    import numpy as np
    before = da[dim].size
    _, index = np.unique(da[dim], return_index=True)
    da = da.isel({dim: index})
    after = da[dim].size
    if verbose:
        print('dropped {} duplicate coord entries.'.format(before-after))
    return da


def groupby_date_xr(da_ts, time_dim='time'):
    df = da_ts[time_dim].to_dataframe()
    df['date'] = df.index.date
    date = df['date'].to_xarray()
    return date


def check_hourly_time_ncfile(file, freq=4, time_dim='time'):
    """
    check for subdaily time coords, and return the missing ones

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    freq : Int, optional
        the daily frequency,4 is 4 X daily. The default is 4.
    time_dim : TYPE, optional
        DESCRIPTION. The default is 'time'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    import xarray as xr
    da_time = xr.open_dataset(file)[time_dim].load()
    dates = groupby_date_xr(da_time)
    counts = da_time.groupby(dates).count().to_dataframe()
    return counts[counts != freq].dropna()


def return_savepath_and_filename_from_filepath(file):
    from pathlib import Path
    filename = file.as_posix().split('/')[-1].split('_')[0:-1]
    filename = '_'.join(filename)
    savepath = Path('/'.join(file.as_posix().split('/')[0:-1]) + '/')
    return savepath, filename


def save_big_ERA5_dataset_as_yearly_files(file, time_dim='time',
                                          verbose=True):
    import xarray as xr
    ds = xr.open_dataset(file)
    years, datasets = zip(*ds.groupby("{}.year".format(time_dim)))
    savepath, filename = return_savepath_and_filename_from_filepath(file)
    paths = [savepath / (filename + '_{}.nc'.format(y)) for y in years]
    # paths = ["%s.nc" % y for y in years]
    if verbose:
        yrmin = min(years)
        yrmax = max(years)
        filemin = filename + '_{}.nc'.format(yrmin)
        filemax = filename + '_{}.nc'.format(yrmax)
        print('saving {} to {}.'.format(filemin, filemax))
    xr.save_mfdataset(datasets, paths)
    return


def replace_nc_yearly_files_list_with_filled(original_files, filled_files, time_dim='time'):
    import xarray as xr
    import numpy as np
    filled_years = [np.unique(xr.open_dataset(x)[time_dim].dt.year).item() for x in filled_files]
    inds = []
    for year in filled_years:
        print('found filled file of year {}'.format(year))
        item_in_ofiles = [x for x in original_files if str(year) in x.as_posix()][0]
        inds.append(original_files.index(item_in_ofiles))
    for ind, filled in zip(inds, filled_files):
        original_files[ind] = filled
    return sorted(original_files)


def reconcat_all_yearly_files(big_file, filled_suffix='_filled', suffix=None,
                              time_dim='time'):
    import xarray as xr
    from dask.diagnostics import ProgressBar
    savepath, filename = return_savepath_and_filename_from_filepath(big_file)
    glob_str = filename + '_' + '[0-9]'*4 + '.nc'
    files = path_glob(savepath, glob_str)
    if filled_suffix is not None:
        glob_str = filename + '_' + '[0-9]'*4 + '{}.nc'.format(filled_suffix)
        filled_files = path_glob(savepath, glob_str)
        files = replace_nc_yearly_files_list_with_filled(files, filled_files, time_dim)
    ds = xr.open_mfdataset(sorted(files))
    ds = ds.sortby(time_dim)
    # dsl = [xr.open_dataset(x, chunks={'time': 1}) for x in sorted(files)]
    # ds = xr.concat(dsl, time_dim)
    years = [x for x in ds.groupby('{}.year'.format(time_dim)).groups.keys()]
    if suffix is not None:
        ds_filename = filename + \
            '_{}-{}_{}.nc'.format(min(years), max(years), suffix)
    else:
        ds_filename = filename + '_{}-{}.nc'.format(min(years), max(years))
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds}
    ds_delayed = ds.to_netcdf(savepath / ds_filename, mode='w',
                              compute=False, encoding=None)
    with ProgressBar():
        results = ds_delayed.compute()
    return


def concat_missing_daily_data_in_yearly_files(missing_data_file, big_file,
                                              time_dim='time'):
    import xarray as xr
    import xesmf as xe
    savepath, filename = return_savepath_and_filename_from_filepath(big_file)
    # regrid missing data coords to big:
    # load missing data file:
    ds = xr.load_dataset(missing_data_file)
    lat_out = xr.open_dataset(big_file)['latitude']
    lon_out = xr.open_dataset(big_file)['longitude']
    ds_out = xr.Dataset({'latitude': (['latitude'], lat_out.values),
                         'longitude': (['longitude'], lon_out.values)})
    regridder = xe.Regridder(ds, ds_out, 'bilinear')
    ds_regrided = regridder(ds)
    # first get the file names of all of the yearly nc files into list:
    glob_str = filename + '_' + '[0-9]'*4 + '.nc'
    files = path_glob(savepath, glob_str)
    # dsl = [xr.open_dataset(x, chunks={time_dim: 1}) for x in sorted(files)]
    # check for missing dates in big file:
    df = check_hourly_time_ncfile(big_file, time_dim=time_dim)
    dates = [x.strftime('%Y-%m-%d') for x in df.index]
    for date in dates:
        print('found missing date : {}'.format(date))
        missing = ds_regrided.sel({time_dim: date})
        year = missing[time_dim].dt.year[0].item()
        # load the correct year from dsl:
        ds_filename = [x for x in files if str(year) in x.as_posix()][0]
        print('found {}.'.format(ds_filename))
        save_concat_date_single_year(ds_filename, missing)
        # ds1 = [x for x in dsl if year in x[time_dim].dt.year.values][0]
        # ds1.load()
    return


def save_concat_date_single_year(big_yearly_filename, missing_ds, time_dim='time', compute=True):
    import xarray as xr
    from dask.diagnostics import ProgressBar
    savepath, filename = return_savepath_and_filename_from_filepath(big_yearly_filename)
    year = missing_ds[time_dim].dt.year[0].item()
    ds1 = xr.open_dataset(big_yearly_filename, chunks={time_dim: 10})
    if compute:
        ds1.persist()
    ds1 = xr.concat([ds1, missing_ds], time_dim)
    ds1 = get_unique_index(ds1, dim=time_dim)
    ds1 = ds1.sortby(time_dim)
    ds1_filename = filename + '_{}_filled.nc'.format(year)
    ds1.to_netcdf(savepath / ds1_filename, 'w', compute=compute)
    if not compute:
        with ProgressBar():
            results = ds1.compute()
    return


def concat_and_save_missing_file(missing_file, big_file, suffix='_1.nc'):
    # very slow...
    import xarray as xr
    from dask.diagnostics import ProgressBar
    ds = xr.open_dataset(big_file, chunks={'time':10000, 'latitude':10, 'longitude':100})
    missing = xr.load_dataset(missing_file)
    dss = xr.concat([ds, missing], 'time')
    dss = get_unique_index(dss)
    dss = dss.sortby('time')
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dss}
    new_file = big_file.as_posix().replace('.nc', suffix)
    dss_delayed = dss.to_netcdf(new_file,
                                'w', encoding=encoding, compute=False)
    with ProgressBar():
        results = dss_delayed.compute()
    return


def read_corrupted_ERA5_table(p_type='reanalysis'):
    from pathlib import Path
    import pandas as pd
    import csv
    cwd = Path().cwd()
    filename = 'era5_corraption_2021-04-15.txt'
    df = pd.read_csv(cwd/filename, delim_whitespace=True, header=0, quoting=3)
    df['dataset'] = df['dataset'].astype(str).str.replace("'", '')
    df['product_type'] = df['product_type'].astype(str).str.replace("'", '')
    df['year'] = df['year'].astype(str).str.replace("'", '')
    df['month'] = df['month'].astype(str).str.replace("'", '')
    df['day'] = df['day'].astype(str).str.replace("'", '')
    df['time'] = df['time'].astype(str).str.replace("'", '')
    df['variable'] = df['variable'].astype(str).str.replace("'", '')
    df['level'] = df['level'].astype(str).str.replace("'", '')
    df1 = df[df['product_type'] == p_type]
    return df1


def download_fixed_ERA5_vars(savepath):
    import cdsapi
    from cds_era5_script import cds_single
    from cds_era5_script import cds_pressure
    df = read_corrupted_ERA5_table('reanalysis')
    # basically, loop over df rows and download each variable:
    for i, row, in df.iterrows():
        var = row['variable']
        if 'single' in row['dataset']:
            data = 'single'
            cds_var = cds_single(var)
        elif 'pressure' in row['dataset']:
            data = 'pressure'
            level = row['level']
            cds_var = cds_pressure(var)
            cds_var.pressure_level = level
        cds_var.year = row['year']
        cds_var.month = row['month']
        cds_var.day = row['day']
        cds_var.time = row['time']
        var = '-'.join(cds_var.variable.split('_'))
        if data == 'pressure':
            filename = 'era5_{}_{}_{}-{}-{}T{}_{}hPa.nc'.format(data, var, cds_var.year, cds_var.month,
                                                                cds_var.day, cds_var.time, cds_var.pressure_level)
        elif data == 'single':
            filename = 'era5_{}_{}_{}-{}-{}T{}.nc'.format(data, var, cds_var.year, cds_var.month,
                                                          cds_var.day, cds_var.time)
        c = cdsapi.Client()
        c.retrieve(name=row['dataset'], request=vars(cds_var),
                   target=savepath / filename)
        print('saved {} to {}'.format(filename, savepath))
    print('Download complete!')
    return


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--savepath', help="a full path to save in the cluster,\
                          e.g., /data11/ziskin/", type=check_path)
    required.add_argument('--mode', help="mode to QA", type=str, choices=['fix_corr'])

    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()
    if args.mode is None:
        print('mode is a required argument, run with -h...')
        sys.exit()
    if args.mode == 'fix_corr':
        download_fixed_ERA5_vars(args.savepath)
