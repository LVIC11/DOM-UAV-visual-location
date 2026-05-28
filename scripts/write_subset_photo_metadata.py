"""从已生成的 photo_metadata.csv 过滤出 03_0015–03_0030 并写入子目录。"""
import csv
import os

in_csv = 'data/Uav-data/query/photo_metadata.csv'
out_dir = 'data/Uav-data/query/subset_15_30'
out_csv = os.path.join(out_dir, 'photo_metadata.csv')

os.makedirs(out_dir, exist_ok=True)

with open(in_csv, newline='') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    rows = [row for row in reader if row and row[0] >= '03_0015.JPG' and row[0] <= '03_0030.JPG']

with open(out_csv, 'w', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(header)
    writer.writerows(rows)

print('Wrote', out_csv, 'with', len(rows), 'rows')
