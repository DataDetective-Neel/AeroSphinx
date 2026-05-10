import netCDF4
import numpy as np

def load_insat_aod(nc_path):
    ds = netCDF4.Dataset(nc_path,'r')
    # pick the AOD variable name dynamically
    var = [v for v in ds.variables if 'AOD' in v.upper()][0]
    aod = ds.variables[var][:].data
    lat = ds.variables['Latitude'][:].data
    lon = ds.variables['Longitude'][:].data
    ds.close()
    return aod, lat, lon
