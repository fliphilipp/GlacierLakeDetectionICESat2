import os
import math
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import h5py

##########################################################################################
def get_size(filename):
    size_bytes = os.path.getsize(filename)
    if size_bytes == 0: return "0 B"
    else:
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        s = round(size_bytes / math.pow(1024, i), 2)
        return "%s %s" % (s, size_name[i])

##########################################################################################
# def convert_time_to_string(dt):
#     epoch = dt + datetime.datetime.timestamp(datetime.datetime(2018,1,1))
#     return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d, %H:%M:%S")

def convert_time_to_string(lake_mean_delta_time): # fixed to match UTC timezone
    # ATLAS SDP epoch is 2018-01-01:T00.00.00.000000 UTC, from ATL03 data dictionary 
    ATLAS_SDP_epoch_datetime = datetime(2018, 1, 1, tzinfo=timezone.utc)
    ATLAS_SDP_epoch_timestamp = datetime.timestamp(ATLAS_SDP_epoch_datetime)
    lake_mean_timestamp = ATLAS_SDP_epoch_timestamp + lake_mean_delta_time
    lake_mean_datetime = datetime.fromtimestamp(lake_mean_timestamp, tz=timezone.utc)
    time_format_out = '%Y-%m-%dT%H:%M:%SZ'
    is2time = datetime.strftime(lake_mean_datetime, time_format_out)
    return is2time

    
##########################################################################################
def encedc(fwnoe='x852\xb4I7\x05\xd5\xff.QB\x18', howjfj='rF\xb4\x9d\t|\xc128\xd2d\xd8uJ_\x9f', nfdoinfrk='misc/test2', jfdsjfds='misc/test1'): 
    import rsa
    with open(nfdoinfrk, 'rb') as jrfonfwlk:
        nwokn = rsa.encrypt(fwnoe.encode(), rsa.PublicKey.load_pkcs1(jrfonfwlk.read()))
        rgnwof = rsa.encrypt(howjfj.encode(), rsa.PublicKey.load_pkcs1(jrfonfwlk.read()))
    with open(jfdsjfds, 'rb') as nwoirlkf:
        rijgorji = rsa.decrypt(nwokn, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()
        napjfpo = rsa.decrypt(rgnwof, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()
    return {'rgnwof':rgnwof, 'nwokn':nwokn, 'napjfpo':napjfpo, 'rijgorji':rijgorji}


##########################################################################################
def decedc(jdfowejpo='1c\x8aR<\xf7jNcP[E\xe1<\x852\xb4I7\x05', jfdsjfds='misc/test1'):
    import rsa
    with open(jfdsjfds, 'rb') as nwoirlkf:
        return rsa.decrypt(jdfowejpo, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()
