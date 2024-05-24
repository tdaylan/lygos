
def cnfg_defa():
    
    jobs = []
    isec = 1
    for i in range(1, 5):
        for j in range(1, 5):
            if i == 4 and  j == 1:
                init(isec, i, j)
                
            #p = multiprocessing.Process(target=work, args=(isec, i, j))
            #jobs.append(p)
            #p.start()


def cnfg_sect():
    
    for isec in range(9, 10):
        for icam in range(4, 5):
            for iccd in range(2, 3):
                init(isec, icam, iccd)



