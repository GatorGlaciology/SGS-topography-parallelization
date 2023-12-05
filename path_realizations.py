import sgs_main
import sys
import os
import re


if __name__ == '__main__':

    # retrieve user parameters
    base_dir = 'Data/Simulated/PIG/'
    out_dir = 'Output/Simulated/'
    loc = 'PIG'

    x = 'x'
    y = 'y'
    z = 'bedrock_altitude (m)'

    for i, file in enumerate(os.listdir(base_dir)):

        print("\n*************************")
        print(f"Starting Simulation {i+1}/12")
        print("*************************\n")

        regex = re.compile(r"^[^_]*_(.*)_[^_]*$")
        type = re.sub(regex, r"\1", file)
        num = re.findall(r'\d+', file)[0]

        in_file = base_dir + file

        out_file = out_dir + loc + "_" + type + "_sim" + num + ".csv"
        out_img = out_dir + "Plots/" + loc + "_" + type + "_plt" + num + ".png"

        sgs_main.run(in_file, out_file, out_img, x, y, z, type)
        
    sys.exit()
