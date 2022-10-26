import pandas as pd
import json
import requests
import datetime


def get_remote_data(day):

    d = day.date().strftime('%Y-%m-%d')

    try:
        r = requests.get('https://rim.met.no/api/v1/observations?sources=SN51610&referenceTime={}T01:00:00Z/{}T21:59:59Z&elements=surface_air_pressure,air_temperature,dew_point_temperature&timeResolution=minutes'.format(d,d),
                verify=False)
    except Exception:
        print("something is fucked")
        return None

    js = json.loads(r.content)

    res = []

    for d in js['data']:
        id_ = d['sourceId']
        time = d['referenceTime']
        for o in d['observations']:
            res.append((id_, o['elementId'], pd.Timestamp(time), o['value'], o['unit']))

    frame = pd.DataFrame(res, columns=('station', 'measurement', 'time', 'value', 'unit'))
    frame = frame.set_index(['station', 'measurement'])

    voss_air_temp = frame.T[('SN51610:0', 'air_temperature')].T.reset_index(drop=True).set_index('time')

    voss_dp_temp = frame.T[('SN51610:0', 'dew_point_temperature')].T.reset_index(drop=True).set_index('time')

    return voss_air_temp, voss_dp_temp


if __name__ == '__main__':

    time = datetime.datetime.today()
    frame = get_remote_data(time)

    frame['value'].plot()
    print(frame)