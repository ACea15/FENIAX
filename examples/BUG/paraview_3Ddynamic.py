# [[file:modelgeneration.org::3Ddynamic][3Ddynamic]]
frames_second = 20
video_length = 6 #seconds
time_interval = int(inp.system.tn / (frames_second * video_length))
import time
time1 = time.time()
rintrinsic, uintrinsic = reconstruction.rbf_based(
        nastran_bdf="./NASTRAN/BUG103.bdf",
        X=config.fem.X,
        time=config.system.t, #range(len(inp.system.t)),
        ra=sol1.dynamicsystem_s1.ra,
        Rab=sol1.dynamicsystem_s1.Cab,
        R0ab=sol1.modes.C0ab,
        vtkpath=inp.driver.sol_path /"paraview/bug",
        plot_timeinterval=time_interval,
        plot_ref=False,
        tolerance=1e-3,
        size_cards=8,
        rbe3s_full=False,
        ra_movie=None)
time2 = time.time()
print(time1-time2)
# 3Ddynamic ends here
