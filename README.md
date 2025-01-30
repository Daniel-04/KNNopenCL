Input files are expected to be comma separated decimals representing features,
last column is class, class column is ignored but expected in testfile.

# Average times with 4MB test and 16MB train inputs:
(10 runs, k=3)
| CPU Cores | Time (seconds) |
|-----------|----------------|
| 1         | 20.9           |
| 2         | 10.5           |
| 4         | 5.3            |
| 8         | 2.7            |

*VERY* old GPU: 2.8 seconds
