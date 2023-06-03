import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# read the CSV files and store the data in pandas dataframes
df_naive = pd.read_csv('CSV/Naive.csv', header=None, names=['Size', 'Time'])
df_tiled = pd.read_csv('CSV/Tiled.csv', header=None, names=['Size', 'Time'])
df_atomics = pd.read_csv('CSV/Atomics.csv', header=None, names=['Size', 'Time'])
df_streams = pd.read_csv('CSV/Streams.csv', header=None, names=['Size', 'Time'])
df_streamed_atomics = pd.read_csv('CSV/StreamedAtomics.csv', header=None, names=['Size', 'Time'])

# calculate the speedup for each program
df_tiled['Speedup'] = df_naive['Time'] / df_tiled['Time']
df_atomics['Speedup'] = df_naive['Time'] / df_atomics['Time']
df_streams['Speedup'] = df_naive['Time'] / df_streams['Time']
df_streamed_atomics['Speedup'] = df_naive['Time'] / df_streamed_atomics['Time']

# plot the graph
plt.plot(df_naive['Size'], df_naive['Time'], label='Naive')
plt.plot(df_tiled['Size'], df_tiled['Time'], label='Tiled')
plt.plot(df_atomics['Size'], df_atomics['Time'], label='Atomics')
plt.plot(df_streams['Size'], df_streams['Time'], label='Streams')
plt.plot(df_streamed_atomics['Size'], df_streamed_atomics['Time'], label='Streamed Atomics')
plt.legend()
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time')
plt.title('Matrix Multiplication Performance Comparison')
plt.savefig('SpeedUpImages/time_comparison.png', bbox_inches='tight') # save the graph

# plot the speedup graph
plt.plot(df_tiled['Size'], df_tiled['Speedup'], label='Tiled')
plt.plot(df_atomics['Size'], df_atomics['Speedup'], label='Atomics')
plt.plot(df_streams['Size'], df_streams['Speedup'], label='Streams')
plt.plot(df_streamed_atomics['Size'], df_streamed_atomics['Speedup'], label='Streamed Atomics')
plt.legend()
plt.xlabel('Matrix Size')
plt.ylabel('Speedup')
plt.title('Matrix Multiplication Performance Comparison')
plt.savefig('SpeedUpImages/speedup_comparison.png', bbox_inches='tight') # save the graph
