import subprocess
import itertools

powers = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
kc = list(range(16, 74, 2))
mc = [1184,1200,1216,1232,1248,1264]
nc = [1920,1936,1952,1968,1984,2000,2016,2032,2048,2064,2080,2096,2112,2128,2144,2160,2176,2192,2208,2224,2240,2256,2272,2288,2304,2320]

insane = [16,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,1040,1056,1072,1088,1104,1120,1136,1152,1168,1184,1200,1216,1232,1248,1264,1280,1296,1312,1328,1344,1360,1376,1392,1408,1424,1440,1456,1472,1488,1504,1520,1536,1552,1568,1584,1600,1616,1632,1648,1664,1680,1696,1712,1728,1744,1760,1776,1792,1808,1824,1840,1856,1872,1888,1904,1920,1936,1952,1968,1984,2000,2016,2032,2048,2064,2080,2096,2112,2128,2144,2160,2176,2192,2208,2224,2240,2256,2272,2288,2304,2320,2336,2352,2368,2384,2400,2416,2432,2448,2464,2480,2496,2512,2528,2544,2560,2576,2592,2608,2624,2640,2656,2672,2688,2704,2720,2736,2752,2768,2784,2800,2816,2832,2848,2864,2880,2896,2912,2928,2944,2960,2976,2992,3008,3024,3040,3056,3072,3088,3104,3120,3136,3152,3168,3184]

cartesian = [p for p in itertools.product(powers, repeat=3)]
cartesian = [(a,b,c) for (a,b,c) in cartesian if a < 256]
cartesian = [(a,b,c) for (a,b,c) in cartesian if b > 32]
cartesian = [(a,b,c) for (a,b,c) in cartesian if c > 64]

dkc = 32
dmc = 1088
dnc = 2176
dmr = 16
dnr = 4

flags = 'MY_OPT=-O4 -DOPENBLAS_SINGLETHREAD -DDGEMM_KC={} -DDGEMM_MC={} -DDGEMM_NC={}'

outfile = open("compileout.txt", "w")

def testkc():
    for testkc in kc:
        filename = "./script_results_kc/kc{}_mc{}_nc{}.txt".format(testkc, dmc, dnc)
        print('testing kc:', testkc)
        subprocess.call(['make', flags.format(testkc, dmc, dnc, dmr, dnr)], stdout=outfile, stderr=outfile)
        subprocess.Popen(['./benchmark-blislab'], stdout=open(filename, 'w')).communicate()

def testmc():
    for testmc in mc:
        filename = "./script_results_mc/kc{}_mc{}_nc{}.txt".format(dkc, testmc, dnc)
        print('testing mc:', testmc)
        subprocess.call(['make', flags.format(dkc, testmc, dnc, dmr, dnr)], stdout=outfile, stderr=outfile)
        subprocess.Popen(['./benchmark-blislab'], stdout=open(filename, 'w')).communicate()

def testnc():
    for testnc in nc:
        filename = "./script_results_nc/kc{}_mc{}_nc{}.txt".format(dkc, dmc, testnc)
        print('testing nc:', testnc)
        subprocess.call(['make', flags.format(dkc, dmc, testnc, dmr, dnr)], stdout=outfile, stderr=outfile)
        subprocess.Popen(['./benchmark-blislab'], stdout=open(filename, 'w')).communicate()

def brute_test():
    total = len(cartesian)
    i = 0
    for test in cartesian:
        filename = "./brute_force_result/kc{}_mc{}_nc{}.txt".format(test[0], test[1], test[2])
        print('curr progress: {} out of {}'.format(i, total))
        subprocess.call(['make', flags.format(test[0], test[1], test[2], dmr, dnr)], stdout=outfile, stderr=outfile)
        subprocess.Popen(['./benchmark-blislab'], stdout=open(filename, 'w')).communicate()
        i += 1


testkc()