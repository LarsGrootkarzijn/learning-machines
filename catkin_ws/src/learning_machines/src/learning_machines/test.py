import numpy as np

red_grid =np.array(([0, 0, 0]
                  ,[0, 0, 0]
                  ,[0, 0, 1]))

print(np.any(red_grid[: ,1] == 1))