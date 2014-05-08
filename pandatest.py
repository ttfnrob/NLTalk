import pandas as pd
import sys

if len(sys.argv) < 2:
  print "No file specificed\n"
else:
  input_filename = sys.argv[1]

df = pd.read_csv(input_filename, skipinitialspace=True, sep='\t')
g =  df.groupby('focus_id')
flist = g['body'].apply(list)

focus_list = []
for k,v in flist.iteritems():
  if (len(v)>5):
    text = ' '.join(v)
    focus_list.insert(0, text)
