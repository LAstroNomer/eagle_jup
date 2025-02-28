import numpy as np
import h5py as h5
import pyread_eagle # type: ignore

def particle_read(ptype,quantity,snapshot,snapfile,
            phys_units=True,
            cgs_units=False):

    # Trim off the "0.hdf5" from our file string so we can loop over the chunks if needed
    trimmed_snapfile = snapfile[:-6]
    # Initialise an index
    snapfile_ind = 0
    while True:
        try:
            with h5.File(trimmed_snapfile+str(snapfile_ind)+'.hdf5', 'r') as f:
                # Grab all the conversion factors from the header and dataset attributes
                h = f['Header'].attrs['HubbleParam']
                a = f['Header'].attrs['ExpansionFactor']
                h_scale_exponent = f['/PartType%i/%s'%((ptype,quantity))].attrs['h-scale-exponent']
                a_scale_exponent = f['/PartType%i/%s'%((ptype,quantity))].attrs['aexp-scale-exponent']
                cgs_conversion_factor = f['/PartType%i/%s'%((ptype,quantity))].attrs['CGSConversionFactor']
        except:
            # If there are no particles of the right type in chunk 0, move to the next one
            print('No particles of type ',ptype,' in snapfile ',snapfile_ind)
            snapfile_ind += 1
            continue
        # If we got what we needed, break from the while loop
        break

    # Load in the quantity
    data_arr = snapshot.read_dataset(ptype,quantity)

    # Cast the data into a numpy array of the correct type
    dt = data_arr.dtype
    data_arr = np.array(data_arr,dtype=dt)

    if np.issubdtype(dt,np.integer):
        # Don't do any corrections if loading in integers
        return data_arr

    else:
        # Do unit corrections, like we did for the catalogues
        if phys_units:
            data_arr *= np.power(h,h_scale_exponent) * np.power(a,a_scale_exponent)

        if cgs_units:
            data_arr = np.array(data_arr,dtype=np.float64)
            data_arr *= cgs_conversion_factor

        return data_arr

