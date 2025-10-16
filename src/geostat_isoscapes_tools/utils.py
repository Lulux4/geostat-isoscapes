import xarray as xr
from xarray import DataArray

def load_xarray_datarray(fn: str, qty : str) -> DataArray:
    """ Just a function to open a netcdf data array """
    with xr.open_dataset(filename_or_obj=fn, engine='netcdf4',decode_times=False) as file:
        dataArray = file[qty]
    return dataArray