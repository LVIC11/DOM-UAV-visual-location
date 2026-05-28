"""从 03.csv 生成 photo_metadata.csv，映射字段到 data/query 的格式。

生成规则（按用户要求）：
- Filename <- filename
- Latitude <- lat
- Longitude <- lon
- Altitude <- height
- Gimball_Roll/Gimball_Yaw/Gimball_Pitch 留空
- Flight_Roll = Omega
- Flight_Yaw = Kappa
- Flight_Pitch = Phi1

已更新映射：
- Flight_Roll = Kappa
- Flight_Yaw = Phi1
- Flight_Pitch = Omega
"""

import csv

in_csv = 'data/Uav-data/query/03.csv'
out_csv = 'data/Uav-data/query/photo_metadata.csv'

with open(in_csv, newline='') as fin, open(out_csv, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow([
        'Filename',
        'Latitude',
        'Longitude',
        'Altitude',
        'Gimball_Roll',
        'Gimball_Yaw',
        'Gimball_Pitch',
        'Flight_Roll',
        'Flight_Yaw',
        'Flight_Pitch',
    ])
    for row in reader:
        filename = row.get('filename')
        lat = row.get('lat')
        lon = row.get('lon')
        height = row.get('height')
        omega = row.get('Omega') or ''
        kappa = row.get('Kappa') or ''
        phi1 = row.get('Phi1') or ''
        # Per user (updated): Flight_Roll=Kappa, Flight_Yaw=Phi1, Flight_Pitch=Omega
        writer.writerow([filename, lat, lon, height, '', '', '', kappa, phi1, omega])

print('Wrote', out_csv)
