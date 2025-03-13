from otoc import OTOC

otoc = OTOC.init(L=4, J=20, hz=1, hx=1, mu='X', nu='Z', i=0, j=3, T=25, tstep=0.1, Dt=20)
otoc.analysis()
otoc.plot_otoc()