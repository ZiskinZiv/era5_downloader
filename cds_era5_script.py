#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:40:34 2019

@author: shlomi
data size scaling estimation: 10 MB for ~120 items
max : 100,000 items ~ 8.3 GB
typical year : 55000 items ~ 4.2 GB (one year, all months, all levels,
 4 times daily, 1.25x1.25 lat/lon grid)
Half year : ~2.1 GB
whole field = 40 years x 2 halves x 2.1 GB = 170.1 GB
time= ~8 hours per half a year, whole field= ~ 27 days
5 fields = 850.5 GB, can parralize to take the same amount of time
"""


def date_range_to_monthly_request_string(
        start_date='1979-01-01', end_date='2019-12-01'):
    import pandas as pd
    from itertools import groupby
    import numpy as np
    """accepts start_date and end_date and parses it to a continous monthly
    seperated ecmwf-api request time string. returns the dates in datetime
    numpy format and a dict of decade as key and corresponding ecmwf date
    request string as value"""
    dates = pd.date_range(start_date, end_date, freq='MS')
    decades = np.array([list(g)
                        for k, g in groupby(dates, lambda i: i.year // 10)])
    dec_value = []
    dec_key = []
    for dec in decades:
        dates_list = []
        decade_dates = dec
        for date in decade_dates:
            formatted_date = date.strftime('%Y%m%d')
            dates_list.append(formatted_date)
        dec_value.append('/'.join(dates_list))
        year = dec[0].year
        dec_key.append(str(10 * divmod(year, 10)[0]))
    return dates, dict(zip(dec_key, dec_value))


class Error(Exception):
    """Base class for other exceptions"""
    pass


class Halfnot1or2(Error):
    """Raised when the half is not 1 or 2"""
    def __init__(self, message):
        self.message = message
    pass


class FieldnotFound(Error):
    """Raised when the field not found in any of the dicts of era5 variable"""
    def __init__(self, message):
        self.message = message
    pass


class era5_variable:
    def __init__(self, era5p1_flag=False, start_year=1979, end_year=2019):
        self.start_year = start_year  # do not change this until era5 publishes more data
        self.end_year = end_year  # era5 reanalysis data lags real time with 2-3 months
        self.pressure = {'phi': 'geopotential',
                         'T': 'temperature',
                         'U': 'u_component_of_wind',
                         'V': 'v_component_of_wind',
                         'omega': 'vertical_velocity',
                         'div': 'divergence',
                         'PV': 'potential_vorticity',
                         'CC': 'fraction_of_cloud_cover',
                         'O3': 'ozone_mass_mixing_ratio',
                         'RH': 'relative_humidity',
                         'VO': 'vorticity',
                         'CIWC': 'specific_cloud_ice_water_content',
                         'Q': 'specific_humidity',
                         'CSWC': 'specific_snow_water_content',
                         'CLWC': 'specific_cloud_liquid_water_content',
                         'CRWC': 'specific_rain_water_content'}
        self.single = {'MSL': 'mean_sea_level_pressure',
                       '2T': '2m_temperature',
                       'E': 'evaporation',
                       'TP': 'total_precipitation',
                       '10WU': '10m_u_component_of_wind',
                       '10WV': '10m_v_component_of_wind',
                       'SP': 'surface_pressure',
                       'TCWV': 'total_column_water_vapour'}
        self.land = {'SN_ALB': 'snow_albedo',
                     'SN_CVR': 'snow_cover',
                     'SN_DWE': 'snow_depth_water_equivalent'}
        self.complete = {'MTTSWR': ['235001', 'Mean temperature tendency' +
                                    ' due to short-wave radiation'],
                         'MTTLWR': ['235002', 'Mean temperature tendency'
                                    + ' due to long-wave radiation'],
                         'MTTPM': ['235005', 'Mean temperature tendency due '
                                   + ' to parametrisations'],

                         'LTI': ['151201', 'Temperature increment from ' +
                                 'relaxation term'],
                         'MUTPM': ['235007', 'Mean eastward wind tendency due to parametrisations'],
                         'T_ML': ['130', 'Temperature from Model Levels'],
                         'U_ML': ['131', 'U component of wind velocity(zonal) from Model Levels'],
                         'V_ML': ['132', 'V component of wind velocity(meridional) from Model Levels'],
                         'Q_ML': ['133', 'Specific humidity from Model Levels']}
        self.era5p1_flag = era5p1_flag
        if era5p1_flag:
            self.start_year = 2000
            self.end_year = 2006
    def list_var(self, var):
        return [x for x in var.keys()]

    def list_years(self):
        import numpy as np
        self.years = np.arange(self.start_year, self.end_year + 1).tolist()
        return self.years

    def list_vars(self):
        single = self.list_var(self.single)
        pressure = self.list_var(self.pressure)
        complete = self.list_var(self.complete)
        land = self.list_var(self.land)
        return single + pressure + complete + land

    def get_model_name(self, field):
        self.field = field
        if field in self.single.keys():
            self.model_name = 'reanalysis-era5-single-levels'
            var = self.single[field]
            print('Single level model selected...')
            return cds_single(var)
        elif field in self.land.keys():
            self.model_name = 'reanalysis-era5-land'
            var = self.land[field]
            print('Land model selected...')
            return cds_single(var)
        elif field in self.pressure.keys():
            self.model_name = 'reanalysis-era5-pressure-levels'
            var = self.pressure[field]
            print('Pressure level model selected...')
            return cds_pressure(var)
        elif field in self.complete.keys():
            if self.era5p1_flag:
                self.model_name = 'reanalysis-era5.1-complete'
                print('ERA5.1 model selected with years set to 2000-2006!')
            else:
                self.model_name = 'reanalysis-era5-complete'
            var = self.complete[field][0]
            print('Complete model selected...working with MARS keywords.')
            return cds_mars(var)
        else:
            raise FieldnotFound('Field not found in any dicts')

    def show_options(self):
        print('The various fields that can be downloaded with this script:')
        self.show('pressure')
        self.show('single')
        self.show('complete')
        self.show('land')

    def show(self, modelname):
        if modelname == 'pressure':
            print('Pressure level fields:')
            for key, value in self.pressure.items():
                print(key + ':', value)
            print('')
        elif modelname == 'land':
            print('Land fields:')
            for key, value in self.land.items():
                print(key + ':', value)
            print('')
        elif modelname == 'single':
            print('Single level fields:')
            for key, value in self.single.items():
                print(key + ':', value)
            print('')
        elif modelname == 'complete':
            print('Complete level fields:')
            for key, value in self.complete.items():
                print(key + ':', value)
            print('')
        else:
            print('model name is invalid!')

    def info(self):
        print('More info about era5: ')
        print('https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis'
              + '-era5-pressure-levels?tab=form')
        print('')

    def desc(self):
        print('ERA5 Varaiable product python script downloader v1.0!')
        print('Author: Shlomi Ziskin Ziv, Atmospheric Sciences Dept., '
              + 'Hebrew University of Jerusalem.')
        print('Email: shlomiziskin@gmail.com')
        print('The following era5 fields can be downloaded and for now, the' +
              ' time period is between ' + str(self.start_year) + ' and '
              + str(self.end_year))
        print('Warning: if you want to download specific year, year option and'
              + ' half option are to be specified togather!')
        self.show_options()
        self.info()


class cds_single:
    def __init__(self, variable):
        self.variable = variable
        self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                      '10', '11', '12']
        self.time = ['00:00', '06:00', '12:00', '18:00']
        self.day = [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31']
        self.year = ['1979', '1980', '1981', '1982', '1983', '1984', '1985',
                     '1986', '1987', '1989']
        self.product_type = 'reanalysis'
        self.grid = [1.25, 1.25]
        self.format = 'netcdf'

    def show(self):
        for key, value in vars(self).items():
            if isinstance(value, list):
                print(key + ':', ', '.join([str(x) for x in value]))
            else:
                print(key + ':', str(value))
        self.count_items()

    def listify_vars(self):
        vars_d = {k: [str(x)] for k, x in vars(self).items() if not
                  isinstance(x, list)}
        vars(self).update(**vars_d)
        return self

    def from_dict(self, d):
        self.__dict__.update(d)
        return self

    def del_attr(self, name):
        delattr(self, name)
        return self

    def count_items(self):
        from numpy import prod
        max_items = 100000
        self.listify_vars()
        attrs_not_to_count = ['product_type', 'grid', 'format']
        count = prod([len(x) for k, x in vars(self).items() if k not in
                      attrs_not_to_count])
        if count >= max_items:
            raise ValueError('Items count is: ' + str(count) +
                             ' and is higher than ' + str(max_items) + ' !')
        print('Items count is: ' + str(count) + ' while max is: ' +
              str(max_items))
        return


class cds_pressure(cds_single):
    def __init__(self, variable):
        cds_single.__init__(self, variable)
        self.pressure_level = [
                    '1', '2', '3',
                    '5', '7', '10',
                    '20', '30', '50',
                    '70', '100', '125',
                    '150', '175', '200',
                    '225', '250', '300',
                    '350', '400', '450',
                    '500', '550', '600',
                    '650', '700', '750',
                    '775', '800', '825',
                    '850', '875', '900',
                    '925', '950', '975',
                    '1000']
        self.year = '1979'

    def select_half(self, half):
            if half == 1:
                self.month = ['01', '02', '03', '04', '05', '06']
            elif half == 2:
                self.month = ['07', '08', '09', '10', '11', '12']
            else:
                raise Halfnot1or2('Half should be 1 or 2...')


class cds_mars:
    def __init__(self, param):
        # self.class = 'ea'
        self.param = param
        self.expver = '1'
        self.stream = 'oper'
        # self.step = '/'.join(['0', '6'])
        self.levtype = 'ml'
        self.levelist = '1/to/137'
        self.time = '00/06/12/18'
        self.grid = [1.25, 1.25]
        self.format = 'netcdf'
        if str(param).startswith('235'):
            self.type = 'fc'
            self.time = '06:00:00/18:00:00'
            self.step = '6/12'
        else:
            self.type = 'an'
    # 'date'    : '2013-01-01',

    def from_dict(self, d):
        self.__dict__.update(d)
        return self

    def del_attr(self, name):
        delattr(self, name)
        return self

    def set_class_atr(self):
        setattr(self, 'class', 'ea')

    def show(self):
        for key, value in vars(self).items():
            if isinstance(value, list):
#                if key == 'date':
#                    print(key + ':', value.split('/')[0] + ' to ' +
#                          value.split('/')[-1])
                print(key + ':', ', '.join([str(x) for x in value]))
            else:
                print(key + ':', str(value))

    def get_date(self, year, half):
        if half == 1:
            self.date = str(year) + '01' + '01' + '/to/' + str(year) + '06' + '31'
        elif half == 2:
            self.date = str(year) + '07' + '01' + '/to/' + str(year) + '12' + '31'
        else:
            raise Halfnot1or2('Half should be 1 or 2...')


#def get_custom_params(custom_fn, cds_obj):
#    import pandas as pd
#    df = pd.read_csv(custom_fn, skiprows=2)
#    dd = dict(zip(df.name.values, df.param.values))
#    c_dict = {}
#    c_dict['filename'] = dd.pop('filename')
#    if dd['stream'] == 'moda':
#        c_dict['monthly'] = True
#    cds_obj.from_dict(dd)
#    cds_obj.del_attr('step')
#    cds_obj.del_attr('time')
#    return cds_obj, c_dict

def get_custom_params(custom_fn, cds_obj):
    import json
    with open(custom_fn) as f:
        dd = json.load(f)
    c_dict = {}
    if 'suffix' in dd.keys():
        c_dict['suffix'] = dd.pop('suffix')
    if 'years' in dd.keys():
        c_dict['years'] = parse_mars_years(dd.pop('years'))
    if 'filename' in dd.keys():
        c_dict['filename'] = dd.pop('filename')
    if 'stream' in dd.keys():
        if dd['stream'] == 'moda':
            c_dict['monthly'] = True
        else:
            c_dict['monthly'] = False
    if 'to_delete' in dd.keys():
        to_delete_list = dd.pop('to_delete')
        for item_to_delete in to_delete_list:
            if item_to_delete in vars(cds_obj).keys():
                print('deleting {}'.format(item_to_delete))
                cds_obj = cds_obj.del_attr(item_to_delete)
    cds_obj.from_dict(dd)
    if 'step' in vars(cds_obj).keys():
        cds_obj = cds_obj.del_attr('step')
        # cds_obj.del_attr('time')
    return cds_obj, c_dict


def generate_filename(modelname, field, cds_obj, half=1, suffix=None):
    """naming method for filenames using field (e.g., 'U', 'T')
    and year and half"""
    if 'single' in modelname.split('-') or 'land' in modelname.split('-'):
        years = '-'.join(str(v) for v in [cds_obj.year[0], cds_obj.year[-1]])
        value_list = ['era5', field, years]
    elif 'pressure' in modelname.split('-'):
        Half = 'H' + str(half)
        if isinstance(cds_obj.year, list):
            year = cds_obj.year[0]
        else:
            year = cds_obj.year
        value_list = ['era5', field, year, Half]
    elif 'complete' in modelname.split('-'):
        Half = 'H' + str(half)
        year = cds_obj.date[:4]
        if 'era5.1' in modelname.split('-'):
            value_list = ['era5p1', field, year, Half]
        else:
            value_list = ['era5', field, year, Half]
    if suffix is not None:
        value_list.append(suffix)
    filename = '_'.join(str(v) for v in value_list) + '.nc'
    return filename


def parse_mars_years(mars_years):
    import numpy as np
    start = mars_years.split('/to/')[0]
    end = mars_years.split('/to/')[-1]
    return np.arange(int(start), int(end) + 1)


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def check_params_file(filepath):
    from pathlib import Path
    filepath = Path(filepath)
    if not filepath.is_file():
        raise argparse.ArgumentTypeError('{} does not exist...'.format(filepath))
    return filepath


def get_decade(start_year, end_year):
    """divide the time legnth into decades, return list of 10 years each"""
    import numpy as np
    all_years = np.arange(int(start_year), int(end_year) + 1)
    yr_chunks = [all_years[x: x+10] for x in range(0, len(all_years), 10)]
    return yr_chunks


def get_era5_field(path, era5_var, cds_obj, c_dict=None, dry=False):
    """downloads the requested era5 fields within a specific year, six months
    (all days, 4x daily), saves it to path.
    available fields are in variable dictionary:"""
    import cdsapi
    import os
    import numpy as np

    def retrieve_era5(c, name, request, target, dry=False):
        if dry:
            print('Dry run! (no download command sent!)')
            return
        else:
            c.retrieve(name=name, request=request, target=target)
            print('Download complete!')
            return

    c = cdsapi.Client()
    modelname = era5_var.model_name
    if c_dict:
        if 'suffix' in c_dict.keys():
            suffix = c_dict['suffix']
        else:
            suffix = None
        if 'filename' in c_dict.keys():
            fn = c_dict['filename']
        else:
            fn = None
        if 'monthly' in c_dict.keys():
            monthly = c_dict.keys()
        else:
            monthly = None
        if 'years' in c_dict.keys():
            user_years = c_dict['years']
        else:
            user_years = None
    else:
        suffix = None
        fn = None
        monthly = None
        user_years = None
    if 'single' in modelname.split('-'):
        if user_years is not None:
            years = get_decade(user_years[0], user_years[-1])
        else:
            years = get_decade(era5_var.start_year, era5_var.end_year)
        for year in years:
            cds_obj.year = year.tolist()
            filename = generate_filename(modelname, era5_var.field, cds_obj,
                                         suffix=suffix)
            if (path / filename).is_file():
                print('{} already exists in {}, skipping...'.format(filename, path))
                continue
            else:
                print('model_name: ' + modelname)
                cds_obj.show()
                print('proccesing request for ' + filename + ' :')
                print('target: {}/{}'.format(path, filename))
                retrieve_era5(c, name=modelname, request=vars(cds_obj),
                              target=path / filename, dry=dry)
                print('')
    elif 'land' in modelname.split('-'):
        # years = get_decade(era5_var.start_year, era5_var.end_year)
        if user_years is not None:
            years = np.arange(2001, user_years[-1] + 1)
        else:
            years = np.arange(2001, era5_var.end_year + 1)
        cds_obj.year = [str(x) for x in years]
        filename = generate_filename(modelname, era5_var.field, cds_obj,
                                     suffix=suffix)
        print('model_name: ' + modelname)
        cds_obj.show()
        print('proccesing request for ' + filename + ' :')
        print('target: {}/{}'.format(path, filename)) 
        # for some strange reason the key 'product_type' kills the session:
        req_dict = vars(cds_obj)
        req_dict.pop('product_type')
        retrieve_era5(c, name=modelname, request=req_dict,
                      target=path / filename, dry=dry)
        print('')
    elif 'pressure' in modelname.split('-'):
        if user_years is not None:
            years = user_years
        else:
            years = era5_var.list_years()
        halves = [1, 2]
        for year in years:
            cds_obj.year = year
            for half in halves:
                cds_obj.select_half(half)
                filename = generate_filename(modelname, era5_var.field,
                                             cds_obj, half, suffix=suffix)
                if (path / filename).is_file():
                    print('{} already exists in {}, skipping...'.format(filename, path))
                    continue
                else:
                    print('model_name: ' + modelname)
                    cds_obj.show()
                    print('proccesing request for ' + filename + ' :')
                    print('target: {}/{}'.format(path, filename))
                    retrieve_era5(c, name=modelname, request=vars(cds_obj),
                                  target=path / filename, dry=dry)
                    print('')
    elif 'complete' in modelname.split('-'):
        if monthly and fn is not None:
            # monthly means
            dt_index, dates_dict = date_range_to_monthly_request_string()
            for decade, mon_dates in dates_dict.items():
                filename = '{}_{}.nc'.format(fn, decade)
                cds_obj.set_class_atr()
                cds_obj.date = mon_dates
                cds_obj.decade = decade
                print('model_name: ' + modelname)
                cds_obj.show()
                print('proccesing request for ' + filename + ' :')
                print('target: {}/{}'.format(path, filename))
                retrieve_era5(c, name=modelname, request=vars(cds_obj),
                              target=path / filename, dry=dry)
        elif monthly is None and fn is not None:
            filename = '{}.nc'.format(fn)
            cds_obj.set_class_atr()
            if 'date' in c_dict.keys():
                cds_obj.date = c_dict['date']
            # cds_obj.decade = decade
            print('model_name: ' + modelname)
            cds_obj.show()
            print('proccesing request for ' + filename + ' :')
            print('target: {}/{}'.format(path, filename))
            retrieve_era5(c, name=modelname, request=vars(cds_obj),
                          target=path / filename, dry=dry)
        elif monthly is None and fn is None:
            if user_years is not None:
                years = user_years
            else:
                years = era5_var.list_years()
            halves = [1, 2]
            cds_obj.set_class_atr()
            for year in years:
                for half in halves:
                    cds_obj.get_date(year, half)
                    filename = generate_filename(modelname, era5_var.field,
                                                 cds_obj, half, suffix=suffix)
                    if (path / filename).is_file():
                        print('{} already exists in {}, skipping...'.format(filename, path))
                        continue
                    else:
                        print('model_name: ' + modelname)
                        cds_obj.show()
                        print('proccesing request for ' + filename + ' :')
                        print('target: {}/{}'.format(path, filename))
                        retrieve_era5(c, name=modelname, request=vars(cds_obj),
                                      target=path / filename, dry=dry)
                        print('')
    return


if __name__ == '__main__':
    import argparse
    import sys
    era5_dummy = era5_variable()
    parser = argparse.ArgumentParser(description=era5_dummy.desc())
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a full path to save in the cluster,\
                          e.g., /data11/ziskin/", type=check_path)
    required.add_argument('--field', help="era5 field abbreviation, e.g., T,\
                          U , V", type=str, choices=era5_dummy.list_vars(),
                          metavar='Era5 Field name abbreviation')
    optional.add_argument('--custom', help='load custom file named \
                          cds_params.txt that contains filename and keywords',
                          type=check_params_file)
    optional.add_argument('--syear', help='start year e.g., 1979', default=1979,
                          type=int)
    optional.add_argument('--eyear', help='end year e.g., 2019', default=2019,
                          type=int)
    optional.add_argument('--era5p1', help='ERA5.1 download flag, automatiacally sets the years to 2000-2006'
                          ,action='store_true')
    optional.add_argument('--dry', help='dry run, no download command sent...'
                          ,action='store_true')

#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.path is None:
        print('path is a required argument, run with -h...')
        sys.exit()
    elif args.field is None:
        print('field is a required argument, run with -h...')
        sys.exit()
    era5_var = era5_variable(start_year=args.syear, end_year=args.eyear, era5p1_flag=args.era5p1)
    cds_obj = era5_var.get_model_name(args.field)
    print('getting era5 all years, field: {}, saving to path: {}'.format(args.field, args.path))
    if args.custom is not None:
        cds_obj, custom_dict = get_custom_params(args.custom, cds_obj)
        get_era5_field(args.path, era5_var, cds_obj, custom_dict, dry=args.dry)
    else:
        get_era5_field(args.path, era5_var, cds_obj, dry=args.dry)
