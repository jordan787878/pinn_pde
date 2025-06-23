# what: combine data from each monte-carlo
# purpose: in order to increase the pdf (monte-carlo) accuracy
# test: by comparing the skew,kurtosis between one monte-carlo pdf vs combined pdf in main.py, we can see that this combine pdf is more accurate
# the data to load is data_[i], where i indicates the i-th run of monte-carlo.
# the combine pdf is saved to DATA_TO_SAVE

import numpy as np
from monte import t1s, visual_pdf

DATA_TO_LOAD_META = "data_"
DATA_TO_SAVE = "data_combine/"
N_data = 3

def main():  

    x1_grid = np.load(DATA_TO_LOAD_META+"1/x1_grid.npy")
    x2_grid = np.load(DATA_TO_LOAD_META+"1/x2_grid.npy")
    np.save(DATA_TO_SAVE+"x1_grid.npy", x1_grid)
    np.save(DATA_TO_SAVE+"x2_grid.npy", x2_grid)
    dx1 = x1_grid[0,1] - x1_grid[0,0]
    dx2 = x2_grid[1,0] - x2_grid[0,0]
    
    for t1 in t1s:
        pdf_combine = x1_grid*0.0
        for j in range(1, N_data+1):
            print("load data: ", DATA_TO_LOAD_META+str(j)+"/pdf_t"+str(t1)+".npy")
            pdf = np.load(DATA_TO_LOAD_META+str(j)+"/pdf_t"+str(t1)+".npy")
            pdf_combine = pdf_combine + pdf/N_data
            p_sum = np.sum(pdf_combine) * dx1 * dx2
            pdf_combine = pdf_combine / p_sum
        # print("check p sum: ", np.sum(pdf_combine) * dx1 * dx2)
        # print(pdf_combine.dtype)
        np.save(DATA_TO_SAVE+"pdf_t"+str(t1)+".npy", pdf_combine)
        visual_pdf(x1_grid, x2_grid, pdf_combine)
    


if __name__ == "__main__":
    main()