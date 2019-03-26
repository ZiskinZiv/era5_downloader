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
Need to add MARS support...
"""


def date_range_to_monthly_request_string(
        start_date='1979-01-01', end_date='2018-12-01'):
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
    def __init__(self):
        self.start_year = 1979  # do not change this until era5 publishes more data
        self.end_year = 2018  # era5 reanalysis data lags real time with 2-3 months
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
                       '10WV': '10m_v_component_of_wind'}
        self.complete = {'MTTSWR': ['235001', 'Mean temperature tendency' +
                                    ' due to short-wave radiation'],
                         'MTTLWR': ['235002', 'Mean temperature tendency' +
                                    ' due to long-wave radiation'],
                         'MTTPM': ['235005', 'Mean temperature tendency due ' +
                                   ' to parametrisations']}

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
        return single + pressure + complete

    def get_model_name(self, field):
        self.field = field
        if field in self.single.keys():
            self.model_name = 'reanalysis-era5-single-levels'
            var = self.single[field]
            print('Single level model selected...')
            return cds_single(var)
        elif field in self.pressure.keys():
            self.model_name = 'reanalysis-era5-pressure-levels'
            var = self.pressure[field]
            print('Pressure level model selected...')
            return cds_pressure(var)
        elif field in self.complete.keys():
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

    def show(self, modelname):
        if modelname == 'pressure':
            print('Pressure level fields:')
            for key, value in self.pressure.items():
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
        self.type = 'fc'
        self.step = '/'.join(['0', '6'])
        self.levtype = 'ml'
        self.levelist = '1/to/137'
        self.time = '06/18'
        self.grid = [1.25, 1.25]
        self.format = 'netcdf'
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


def get_custom_params(custom_fn, cds_obj):
    import pandas as pd
    df = pd.read_csv(custom_fn, skiprows=2)
    dd = dict(zip(df.name.values, df.param.values))
    c_dict = {}
    c_dict['filename'] = dd.pop('filename')
    if dd['stream'] == 'moda':
        c_dict['monthly'] = True
    cds_obj.from_dict(dd)
    cds_obj.del_attr('step')
    cds_obj.del_attr('time')
    return cds_obj, c_dict


def generate_filename(modelname, field, cds_obj, half=1):
    """naming method for filenames using field (e.g., 'U', 'T')
    and year and half"""
    if 'single' in modelname.split('-'):
        years = '-'.join(str(v) for v in [cds_obj.year[0], cds_obj.year[-1]])
        value_list = ['era5', field, years]
    elif 'pressure' in modelname.split('-'):
        Half = 'H' + str(half)
        year = cds_obj.year
        value_list = ['era5', field, year, Half]
    elif 'complete' in modelname.split('-'):
        Half = 'H' + str(half)
        year = cds_obj.date[:4]
        value_list = ['era5', field, year, Half]
    filename = '_'.join(str(v) for v in value_list) + '.nc'
    return filename


def check_path(path):
    import os
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return path


def check_params_file(fn):
    import os
    path_fn = os.getcwd() + '/' + fn
    if not os.path.isfile(path_fn):
        raise argparse.ArgumentTypeError(path_fn + ' does not exist...')
    return path_fn


def get_decade(start_year, end_year):
    """divide the time legnth into decades, return list of 10 years each"""
    import numpy as np
    all_years = np.arange(int(start_year), int(end_year) + 1)
    yr_chunks = [all_years[x: x+10] for x in range(0, len(all_years), 10)]
    return yr_chunks


def get_era5_field(path, era5_var, cds_obj, c_dict=None):
    """downloads the requested era5 fields within a specific year, six months
    (all days, 4x daily), saves it to path.
    available fields are in variable dictionary:"""
    def retrieve_era5(c, name, request, target):
        c.retrieve(name=name, request=request, target=target)
        print('Download complete!')
        return

    import cdsapi
    import os
    c = cdsapi.Client()
    modelname = era5_var.model_name
    if 'single' in modelname.split('-'):
        years = get_decade(era5_var.start_year, era5_var.end_year)
        for year in years:
            cds_obj.year = year.tolist()
            filename = generate_filename(modelname, era5_var.field, cds_obj)
            if os.path.isfile(os.path.join(path, filename)):
                print(filename + ' already exists in ' + path +
                      ' skipping...')
                continue
            else:
                print('model_name: ' + modelname)
                cds_obj.show()
                print('proccesing request for ' + filename + ' :')
                print('target: ' + path + filename)
                retrieve_era5(c, name=modelname, request=vars(cds_obj),
                              target=path + filename)
                print('')
    elif 'pressure' in modelname.split('-'):
        halves = [1, 2]
        years = era5_var.list_years()
        for year in years:
            cds_obj.year = year
            for half in halves:
                cds_obj.select_half(half)
                filename = generate_filename(modelname, era5_var.field,
                                             cds_obj, half)
                if os.path.isfile(os.path.join(path, filename)):
                    print(filename + ' already exists in ' + path +
                          ' skipping...')
                    continue
                else:
                    print('model_name: ' + modelname)
                    cds_obj.show()
                    print('proccesing request for ' + filename + ' :')
                    print('target: ' + path + filename)
                    retrieve_era5(c, name=modelname, request=vars(cds_obj),
                                  target=path + filename)
                    print('')
    elif 'complete' in modelname.split('-'):
        if not c_dict:
            halves = [1, 2]
            years = era5_var.list_years()
            cds_obj.set_class_atr()
            for year in years:
                for half in halves:
                    cds_obj.get_date(year, half)
                    filename = generate_filename(modelname, era5_var.field,
                                                 cds_obj, half)
                    if os.path.isfile(os.path.join(path, filename)):
                        print(filename + ' already exists in ' + path +
                              ' skipping...')
                        continue
                    else:
                        print('model_name: ' + modelname)
                        cds_obj.show()
                        print('proccesing request for ' + filename + ' :')
                        print('target: ' + path + filename)
                        retrieve_era5(c, name=modelname, request=vars(cds_obj),
                                      target=path + filename)
                        print('')
        else:
            # custom mode: relay on user to get everything, custom is the filename
            if c_dict['monthly']:
                # monthly means
                dt_index, dates_dict = date_range_to_monthly_request_string()
                for decade, mon_dates in dates_dict.items():
                    filename = c_dict['filename'] + '_' + decade + '.nc'
                    cds_obj.set_class_atr()
                    cds_obj.date = mon_dates
                    cds_obj.decade = decade
                    print('model_name: ' + modelname)
                    cds_obj.show()
                    print('proccesing request for ' + filename + ' :')
                    print('target: ' + path + filename)
                    retrieve_era5(c, name=modelname, request=vars(cds_obj),
                                  target=path + filename)
    return


if __name__ == '__main__':
    import argparse
    import sys
    era5_var = era5_variable()
    era5_var.start_year = 1979
    era5_var.end_year = 2018
    parser = argparse.ArgumentParser(description=era5_var.desc())
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a full path to save in the cluster,\
                          e.g., /data11/ziskin/", type=check_path)
    required.add_argument('--field', help="era5 field abbreviation, e.g., T,\
                          U , V", type=str, choices=era5_var.list_vars(),
                          metavar='Era5 Field name abbreviation')
    optional.add_argument('--custom', help='load custom file named \
                          cds_params.txt that contains filename and keywords'
                          , type=check_params_file)
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
    cds_obj = era5_var.get_model_name(args.field)
    print('getting era5 all years, field: ' + args.field + ', saving to path:'
          + args.path)
    if args.custom is not None:
        cds_obj, custom_dict = get_custom_params(args.custom, cds_obj)
        get_era5_field(args.path, era5_var, cds_obj, custom_dict)
    else:
        get_era5_field(args.path, era5_var, cds_obj)
