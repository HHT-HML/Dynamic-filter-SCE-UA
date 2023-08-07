#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
# Purpose:
# Dependencies: 	numpy
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

## Refer to paper:
##  'EFFECTIVE AND EFFICIENT GLOBAL OPTIMIZATION FOR CONCEPTUAL
##  RAINFALL-RUNOFF MODELS' BY DUAN, Q., S. SOROOSHIAN, AND V.K. GUPTA,
##  WATER RESOURCES RESEARCH, VOL 28(4), PP.1015-1031, 1992.
##
from SALib.test_functions import Sobol_G
import random
import numpy as np
from SALib.analyze import morris
from SALib.sample.morris import sample

from SCE_functioncall import *

def cceua(values,wq_inst,Y,s,sf,new_add_anlnyse,new_add_romver,bl,bu,bl_old,bu_old,icall,maxn,iseed,none_anlnyse,x_romver,none_anlnyse_old,x_romver_old,flag,testcase=True):
#   This is the subroutine for generating a new point in a simplex
#
#   s(.,.) = the sorted simplex in order of increasing function values
#   s(.) = function values in increasing order
#
# LIST OF LOCAL VARIABLES
#   sb(.) = the best point of the simplex
#   sw(.) = the worst point of the simplex
#   w2(.) = the second worst point of the simplex
#   fw = function value of the worst point
#   ce(.) = the centroid of the simplex excluding wo
#   snew(.) = new point generated from the simplex
#   iviol = flag indicating if constraints are violated
#         = 1 , yes
#         = 0 , no
    Flag=True
    if(flag):
        s = list(s)
        for i in range(len(s)):
            s[i]=low_demon(s[i],new_add_anlnyse)
        s=np.array(s)
    else:
        Flag=False
    nps,nopt=s.shape
    n = nps
    m = nopt
    alpha = 1.0
    beta = 0.5

    # Assign the best and worst points:
    sb=s[0,:]
    fb=sf[0]
    sw=s[-1,:]
    fw=sf[-1]

    # Compute the centroid of the simplex excluding the worst point:
    # 计算单纯形的质心，排除最差点
    ce= np.mean(s[:-1,:],axis=0)

    # Attempt a reflection point
    snew = ce + alpha*(ce-sw)

    if(len(snew)==15):
        print("出错啦！-----1")

    # Check if is outside the bounds:
    ibound=0
    if(Flag):
        s1 = snew - bl
    else:
        s1=snew-bl_old
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=1
    if(Flag):
        s1 = bu - snew
    else:
        s1 = bu_old - snew
    idx=(s1<0).nonzero()
    if idx[0].size != 0:
        ibound=2

    if ibound >= 1:
        flag=False
        snew = SampleInputMatrix(1,len(bu_old),bu_old,bl_old,iseed,distname='randomUniform')[0]  #checken!!


    if (len(snew) == 15):
        print("出错啦！-----2")
    if(flag & Flag):
        print("到特殊情况了！")
##    fnew = functn(nopt,snew);
    #把新产生的解放入morris中的老values中
    x_new=list(snew)
    if(Flag):
        if (flag):
            if (len(none_anlnyse) > 0):
                x_new = list(recover_demon(x_new, none_anlnyse, x_romver))
        else:
            if (len(none_anlnyse_old) > 0):
                x_new = list(recover_demon(x_new, none_anlnyse_old, x_romver_old))
    else:
        x_new = list(recover_demon(x_new, none_anlnyse_old, x_romver_old))
    #values.append(x_new)
    if(Flag):
        if (flag):
            fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse, x_romver, testcase=testcase)
        else:
            fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
    else:
        fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
    icall += 1

    # Reflection failed; now attempt a contraction point:
    if fnew > fw:
        snew = sw + beta*(ce-sw)
        if (len(snew) == 15):
            print("出错啦！-----3")
        #x_new = list(snew)
        if(Flag):
            if (flag):
                if (len(none_anlnyse) > 0):
                    x_new = list(recover_demon(list(snew), none_anlnyse, x_romver))
                    fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse, x_romver, testcase=testcase)
            else:
                if (len(none_anlnyse_old) > 0):
                    if (len(x_new) == len(snew)):
                        x_new = list(recover_demon(list(snew), none_anlnyse_old, x_romver_old))
                        fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
                    else:
                        x_new = list(recover_demon(list(snew), none_anlnyse, x_romver))
                        fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse, x_romver, testcase=testcase)
                        snew = np.array(recover_demon(list(snew), new_add_anlnyse, new_add_romver))
                        if (len(snew) == 15):
                            print("出错啦！-----4")
        else:
            x_new = list(recover_demon(list(snew), none_anlnyse_old, x_romver_old))
            fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
        #values.append(x_new)
        icall += 1

    # Both reflection and contraction have failed, attempt a random point;
        if fnew > fw:
            flag=False
            snew = SampleInputMatrix(1,len(bu_old),bu_old,bl_old,iseed,distname='randomUniform')[0]  #checken!!
            if (len(snew) == 15):
                print("出错啦！-----5")
            x_new = list(snew)
            if(Flag):
                if (flag):
                    if (len(none_anlnyse) > 0):
                        x_new = list(recover_demon(x_new, none_anlnyse, x_romver))
                else:
                    if (len(none_anlnyse_old) > 0):
                        x_new = list(recover_demon(x_new, none_anlnyse_old, x_romver_old))
                if (flag):
                    fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse, x_romver, testcase=testcase)
                else:
                    fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
            else:
                x_new = list(recover_demon(x_new, none_anlnyse_old, x_romver_old))
                fnew = EvalObjF(nopt, snew, wq_inst, Y, none_anlnyse_old, x_romver_old, testcase=testcase)
            #values.append(x_new)
            icall += 1
    if(Flag):
        if (flag):
            snew = list(snew)
            snew = recover_demon(snew, new_add_anlnyse, new_add_romver)
            if (len(snew) == 15):
                print("出错啦！-----6")
            snew = np.array(snew)
    # END OF CCE
    return snew,fnew,icall,flag


def sceua(values,wq_inst,Y,x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,problem,testcase=True):
# This is the subroutine implementing the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S.2011
#
# Definition:
#  x0 = the initial parameter array at the start; np.array
#     = the optimized parameter array at the end;
#  f0 = the objective function value corresponding to the initial parameters
#     = the objective function value corresponding to the optimized parameters
#  bl = the lower bound of the parameters; np.array
#  bu = the upper bound of the parameters; np.array
#  iseed = the random seed number (for repetetive testing purpose)
#  iniflg = flag for initial parameter array (=1, included it in initial
#           population; otherwise, not included)
#  ngs = number of complexes (sub-populations)
#  npg = number of members in a complex
#  nps = number of members in a simplex
#  nspl = number of evolution steps for each complex before shuffling
#  mings = minimum number of complexes required during the optimization process
#  maxn = maximum number of function evaluations allowed during optimization
#  kstop = maximum number of evolution loops before convergency
#  percento = the percentage change allowed in kstop loops before convergency

# LIST OF LOCAL VARIABLES
#    x(.,.) = coordinates of points in the population
#    xf(.) = function values of x(.,.)
#    xx(.) = coordinates of a single point in x
#    cx(.,.) = coordinates of points in a complex
#    cf(.) = function values of cx(.,.)
#    s(.,.) = coordinates of points in the current simplex
#    sf(.) = function values of s(.,.)
#    bestx(.) = best point at current shuffling loop
#    bestf = function value of bestx(.)
#    worstx(.) = worst point at current shuffling loop
#    worstf = function value of worstx(.)
#    xnstd(.) = standard deviation of parameters in the population
#    gnrng = normalized geometric mean of parameter ranges
#    lcs(.) = indices locating position of s(.,.) in x(.,.)
#    bound(.) = bound on ith variable being optimized
#    ngs1 = number of complexes in current population
#    ngs2 = number of complexes in last population
#    iseed1 = current random seed
#    criter(.) = vector containing the best criterion values of the last
#                10 shuffling loops

    # Initialize SCE parameters:
    nopt=x0.size
    npg=2*nopt+1
    nps=nopt+1
    nspl=npg
    mings=ngs
    npt=npg*ngs
    none_anlnyse=[]
    bound = bu-bl  #np.array

    # Create an initial population to fill array x(npt,nopt):
    #x = SampleInputMatrix(npt,nopt,bu,bl,iseed,distname='randomUniform')
    inital_count=int(npt/(problem["num_vars"]+1))
    x = sample(problem, N=inital_count, num_levels=4, optimal_trajectories=None)
    inital_res=inital_count*(problem["num_vars"]+1)
    if iniflg==1:
        x[0, :] = x0

    nloop=0
    icall=0
    xf=np.zeros(inital_res)
    x_romver=[]
    for i in range (inital_res):
        x_new=list(x[i, :])
        if(len(none_anlnyse)>0):
            x_new=list(recover_demon(x_new,none_anlnyse,x_romver))
        values.append(x_new)
        xf[i] = EvalObjF(nopt, x[i, :],wq_inst,Y,none_anlnyse,x_romver, testcase=testcase)
        Y.append(xf[i])
        icall += 1
    f0 = xf[0]
    while len(Y)%(problem["num_vars"]+1)!=0:
        Y.append(Y[len(Y)-1])
        values.append(values[len(values)-1])
    Y1=np.array(Y)
    values1=np.array(values)
    Si = morris.analyze(
        problem,
        values1,
        Y1,
        conf_level=0.95,
        print_to_console=True,
        num_levels=4,
        num_resamples=100,
    )
    si_path=os.getcwd()+os.sep+"Si_inital.txt"
    si_file=open(si_path,"w+")
    for i in range(len(Si["mu"])):
        si_file.write(str(Si["mu"][i])+str('    ')+str(Si["mu_star"][i])+'\n')
    si_file.write(str("-----------------------------------------") + '\n')
    si_file.close()
    #for i in range(nopt):
    #    if(Si["mu"][i]==0):
    #        for j in range(len(x)):
   #             x[j,i]=x[1,i]

    # Sort the population in order of increasing function values;
    idx = np.argsort(xf)
    xf = np.sort(xf)
    x=x[idx,:]

    # Record the best and worst points;
    bestx=x[0,:]
    bestf=xf[0]
    worstx=x[-1,:]
    worstf=xf[-1]

    BESTF=bestf
    BESTX=bestx
    ICALL=icall

    # Compute the standard deviation for each parameter
    xnstd=np.std(x,axis=0)

    # Computes the normalized geometric range of the parameters
    gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

    print('The Initial Loop: 0')
    print(' BESTF:  %f ' %bestf)
    print(' BESTX:  ')
    print(bestx)
    print(' WORSTF:  %f ' %worstf)
    print(' WORSTX: ')
    print(worstx)
    print('     ')

    filepath=os.getcwd()
    result_path=filepath+os.sep+"result_20.txt"
    result_file=open(result_path,"w+")
    result_file.write(str('The Initial Loop: 0')+str('\n'))
    result_file.write(str(' BESTF:  %f ' %bestf)+str("\n"))
    result_file.write(str(' BESTX:  ')+str("\n"))
    result_file.write(str(bestx)+'\n')
    result_file.write(str(' WORSTF:  %f ' %worstf)+str("\n"))
    result_file.write(str('  WORSTX: ')+str("\n"))
    result_file.write(str(worstx)+'\n')
    result_file.write('                    '+'\n')
    # Check for convergency;
    if icall >= maxn:
        print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
        print('ON THE MAXIMUM NUMBER OF TRIALS ')
        print(maxn)
        print('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
        print(icall)
        print('OF THE INITIAL LOOP!')
        return 0

    if gnrng < peps:
        print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
        return 0

    # Begin evolution loops:
    nloop = 0
    criter=[]
    criter_change=1e+5

    #挑选出无敏感性的参数，降低参数率定的维度：
    si_all=0
    for i in range(nopt):
        si_all+=float(Si["mu_star"][i])
    for i in range(nopt):
        if(float(Si["mu_star"][i])<=0.1):
            none_anlnyse.append(i)
    nopt_old = nopt-len(none_anlnyse)
    npg_old = 2 * nopt_old + 1
    nps_old = nopt_old + 1
    nspl_old = npg_old
    npt_old = npg_old * ngs
    if(len(none_anlnyse)>0):
        x=list(x)
        for i in range(len(none_anlnyse)):
            row=0
            x_romver.append(x[0][none_anlnyse[i]-i])
            for l in x:
                rest=list(l[:none_anlnyse[i]-i])
                rest.extend(l[none_anlnyse[i]-i+1:])
                x[row]=rest
                row+=1
        x_temp=[]
        xf_temp=[]
        count=0
        for i in x:
            if i not in x_temp:
                x_temp.append(i)
                xf_temp.append(xf[count])
            count+=1
        while(len(x_temp)<npg_old * ngs):
            x_temp.append(x_temp[len(x_temp)-1])
            xf_temp.append(xf_temp[len(xf_temp)-1])
        x.clear()
        xf=list(xf)
        xf.clear()
        xf=xf_temp
        x=x_temp
        x=np.array(x)
        xf=np.array(xf)
    bl = low_demon(bl, none_anlnyse)
    bu=low_demon(bu,none_anlnyse)
    bound=low_demon(bound,none_anlnyse)
    while icall<maxn and gnrng>peps and criter_change>pcento:
        nloop+=1
        cx_total = np.zeros((npg_old * ngs, nopt_old))
        cf_total = np.zeros(npg_old * ngs)
        for igs in range(ngs):
            k1=np.array(range(npg_old))
            k2=k1*ngs+igs

            cx_total[k1+npg_old*igs, :] = x[k2, :]
            cf_total[k1+npg_old*igs] = xf[k2]
        idx_new = np.argsort(cf_total)
        cf_total = np.sort(cf_total)
        cx_total = cx_total[idx_new, :]
        x_romver_old=[]
        for i in range(len(x_romver)):
            x_romver_old.append(x_romver[i])
        none_anlnyse_old=[]
        for i in range(len(none_anlnyse)):
            none_anlnyse_old.append(none_anlnyse[i])
        sw = cx_total[-1, :]
        # 计算单纯形的质心，排除最差点
        sb=cx_total[0,:]
        ce = np.mean(cx_total[:-1, :], axis=0)
        ce+=ce-sw
        #判断新的范围是否越界
        for i in range(len(ce)):
            if(ce[i]<bl[i]):
                ce[i]=bl[i]
            elif(ce[i]>bu[i]):
                ce[i]=bu[i]
        x_new_add,xf_new_add,new_anlnyse,x_romver_new,new_add_anlnyse,new_add_romver=new_none_anlnyse(ce, sw, none_anlnyse_old, wq_inst, nopt,x_romver_old,testcase)
        for i in range(len(x_new_add)):
            values.append(recover_demon(x_new_add[i], none_anlnyse, x_romver))
            Y.append(xf_new_add[i])
        #去除降低参数个数之后参数中的重复值
        if(len(new_add_anlnyse)>0):
            x_new_temp_old=[]
            xf_new_temp_old=[]
            x_new_add=list(x_new_add)
            count=0
            for i in x_new_add:
                x_new_temp_old.append(i)
                xf_new_temp_old.append(xf_new_add[count])
                count+=1
            x_new_temp_old1=[]
            for i in x_new_temp_old:
                x_new_temp_old1.append(low_demon(i,new_add_anlnyse))
            x_new_temp_old.clear()
            x_new_temp_old=x_new_temp_old1
            x_new_temp_new=[]
            xf_new_temp_real=[]
            x_new_temp_real=[]
            count=0
            for i in x_new_temp_old:
                if list(i) not in x_new_temp_new:
                    x_new_temp_new.append(list(i))
                    x_new_temp_real.append(x_new_add[count])
                    xf_new_temp_real.append(xf_new_add[count])
                count+=1
            xf_new_add=list(xf_new_add)
            x_new_add.clear()
            xf_new_add.clear()
            x_new_add=np.array(x_new_temp_real)
            xf_new_add=np.array(xf_new_temp_real)

        x=list(x)
        xf=list(xf)
        for i in range(len(x_new_add)):
            x.append(x_new_add[i])
            xf.append(xf_new_add[i])
        x=np.array(x)
        xf=np.array(xf)
        # Sort the population in order of increasing function values;
        idx = np.argsort(xf)
        xf = np.sort(xf)
        x = x[idx, :]

        nopt1 = nopt - len(new_anlnyse)
        if(nopt1<=0):
            continue
        npg1 = 2 * nopt1 + 1
        nps1 = nopt1 + 1
        nspl1 = npg1
        npt1 = npg1 * ngs
        if(len(new_anlnyse)>len(none_anlnyse)):
            bl_new = low_demon(bl, new_add_anlnyse)
            bu_new = low_demon(bu, new_add_anlnyse)
            bound_new=low_demon(bound, new_add_anlnyse)
        else:
            bl_new=bl
            bu_new=bu
            bound_new=bound
        x_new=[]
        for i in x:
            x_new.append(i)
        if (len(new_anlnyse) > len(none_anlnyse)):
            for i in range(len(x_new)):
                x_new[i] = recover_demon(x_new[i], none_anlnyse, x_romver)
            x_new = list(x_new)
            for i in range(len(new_anlnyse)):
                row = 0
                for l in x_new:
                    rest = list(l[:new_anlnyse[i] - i])
                    rest.extend(l[new_anlnyse[i] - i + 1:])
                    x_new[row] = rest
                    row += 1
            for i in range(len(x_new)):
                x_new[i]=recover_demon(x_new[i],new_add_anlnyse,new_add_romver)
            #给不具备敏感性的参数附上相同的值

        x_new=np.array(x_new)
        # Loop on complexes (sub-populations);
        for igs in range(ngs):
            # Partition the population into complexes (sub-populations);
            flag=True
            cx=np.zeros((npg1,nopt_old))
            cf=np.zeros((npg1))

            k1=np.array(range(npg1))
            k2=k1*ngs+igs

            cx[k1, :] = x_new[k2, :]
            cf[k1] = xf[k2]

            # Evolve sub-population igs for nspl steps:
            for loop in range(nspl1):

                # Select simplex by sampling the complex according to a linear
                # probability distribution

                lcs=np.array([0]*nps1)
                lcs[0] = 1
                for k3 in range(1,nps1):
                    for i in range(1000):
##                        lpos = 1 + int(np.floor(npg+0.5-np.sqrt((npg+0.5)**2 - npg*(npg+1)*random.random())))
                        lpos = int(np.floor(npg1+0.5-np.sqrt((npg1+0.5)**2 - npg1*(npg1+1)*random.random())))
##                        idx=find(lcs(1:k3-1)==lpos)
                        idx=(lcs[0:k3]==lpos).nonzero()  #check of element al eens gekozen
                        if idx[0].size == 0:
                            break

                    lcs[k3] = lpos
                lcs.sort()

                # Construct the simplex:
                s = np.zeros((nps1,nopt_old))
                s=cx[lcs,:]
                sf = cf[lcs]

                snew,fnew,icall,flag=cceua(values,wq_inst,Y,s,sf,new_add_anlnyse,new_add_romver,bl_new,bu_new,bl,bu,icall,maxn,iseed,new_anlnyse,x_romver_new,none_anlnyse,x_romver,flag,testcase=testcase)

                # Replace the worst point in Simplex with the new point:
                s[-1,:] = snew
                sf[-1] = fnew

                # Replace the simplex into the complex;
                cx[lcs,:] = s
                cf[lcs] = sf

                # Sort the complex;
                idx = np.argsort(cf)
                cf = np.sort(cf)
                cx=cx[idx,:]

            # End of Inner Loop for Competitive Evolution of Simplexes
            #end of Evolve sub-population igs for nspl steps:

            # Replace the complex back into the population;
            #cx=list(cx)
            #for i in range(len(cx)):
                #cx[i]=recover_demon(cx[i],new_add_anlnyse,new_add_romver)
            #cx=np.array(cx)
            x[k2,:] = cx[k1,:]
            xf[k2] = cf[k1]

        # End of Loop on Complex Evolution;
        # Shuffled the complexes;
        idx = np.argsort(xf)
        xf = np.sort(xf)
        x=x[idx,:]

        PX=x
        PF=xf

        # Record the best and worst points;
        bestx=x[0,:]
        bestf=xf[0]
        worstx=x[-1,:]
        worstf=xf[-1]

        if(len(none_anlnyse)>0):
            bestx=np.array(recover_demon(bestx,none_anlnyse,x_romver))
            worstx=np.array(recover_demon(worstx, none_anlnyse,x_romver))
        BESTX = np.append(BESTX,bestx, axis=0) #appenden en op einde reshapen!!
        BESTF = np.append(BESTF,bestf)
        ICALL = np.append(ICALL,icall)

        # Compute the standard deviation for each parameter
        #for i in range(nopt):
        #    if (Si["mu"][i] == 0):
        #        for j in len(x):
       #             x[j, i] = x[1, i]
        xnstd=np.std(x,axis=0)

        # Computes the normalized geometric range of the parameters
        gnrng=np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bound)))

        print('Evolution Loop: %d  - Trial - %d' %(nloop,icall))
        print(' BESTF:  %f ' %bestf)
        print(' BESTX:  ')
        print(bestx)
        print(' WORSTF:  %f ' %worstf)
        print(' WORSTX: ')
        print(worstx)
        print('     ')
        result_file.write(str('Evolution Loop: %d  - Trial - %d' %(nloop,icall))+'\n')
        result_file.write(str(' BESTF:  %f ' %bestf)+'\n')
        result_file.write(str(' BESTX:  ')+'\n')
        result_file.write(str(bestx))
        result_file.write(str(' WORSTF:  %f ' %worstf)+'\n')
        result_file.write(str(' WORSTX: ')+'\n')
        result_file.write(str(worstx))
        result_file.write(str('                                 ')+'\n')
        # Check for convergency;
        if icall >= maxn:
            print('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            print('ON THE MAXIMUM NUMBER OF TRIALS ')
            print(maxn)
            print('HAS BEEN EXCEEDED.')

        if gnrng < peps:
            print('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')

        criter=np.append(criter,bestf)

        if nloop >= kstop: #nodig zodat minimum zoveel doorlopen worden
            criter_change= np.abs(criter[nloop-1]-criter[nloop-kstop])*100
            criter_change= criter_change/np.mean(np.abs(criter[nloop-kstop:nloop]))
            if criter_change < pcento:
                print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY LESS THAN THE THRESHOLD %f' %(kstop,pcento))
                print('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')

    # End of the Outer Loops
    print('SEARCH WAS STOPPED AT TRIAL NUMBER: %d' %icall)
    print('NORMALIZED GEOMETRIC RANGE = %f'  %gnrng)
    print('THE BEST POINT HAS IMPROVED IN LAST %d LOOPS BY %f' %(kstop,criter_change))
    result_file.close()
    #reshape BESTX
    #将不具备敏感性的各参数放入其中：

    BESTX = BESTX.reshape(int(BESTX.size / nopt), nopt)
    #bestx=np.array(recover_demon(bestx,none_anlnyse))

    # END of Subroutine sceua
    return bestx,bestf,BESTX,BESTF,ICALL
def low_demon_mutil(x,none_anlnyse):
    if (len(none_anlnyse) > 0):
        for i in range(len(none_anlnyse)):
            row=0
            for l in x:
                rest = list(l[:none_anlnyse[i] - i])
                rest.extend(l[none_anlnyse[i] - i + 1:])
                x[row]=rest
                row+=1
    return x
def low_demon(x,none_anlnyse):
    if(len(none_anlnyse)>0):
        x=list(x)
        for i in range(len(none_anlnyse)):
            rest=list(x[:none_anlnyse[i]-i])
            rest.extend(x[none_anlnyse[i] - i + 1:])
            x=rest
    return np.array(x)
def recover_demon_mutily(x,none_anlnyse,x_romver):
    if (len(none_anlnyse) > 0):
        x = list(x)
        for i in range(len(none_anlnyse)):
            row=0
            for l in x:
                rest = list(l[:none_anlnyse[i]])
                rest.append(x_romver[i])
                rest.extend(l[none_anlnyse[i]:])
                x[row] = rest
        x = np.array(x)
    return x
def recover_demon(x,none_anlnyse,x_romver):
    if (len(none_anlnyse) > 0):
        x = list(x)
        for i in range(len(none_anlnyse)):
            rest = list(x[:none_anlnyse[i]])
            rest.append(x_romver[i])
            rest.extend(x[none_anlnyse[i]:])
            x = rest
    return x
def new_sample_x(x1,x2,ngs,nspl):
    result = []
    problem = {
        'num_vars': 4,
        'names': [],
        'groups': None,
        'bounds': []}
    problem['num_vars'] = len(x1)
    for i in range(len(x1)):
        k=i+1
        problem['names'].append(str("x")+str(k))
        if(x1[i]<=x2[i]):
            problem['bounds'].append([x1[i],x2[i]])
        else:
            problem['bounds'].append([x2[i], x1[i]])
    x = sample(problem, N=int((ngs*nspl)/(len(x1)+1)), num_levels=4, optimal_trajectories=None)
    return x
def new_none_anlnyse(snew,sw,none_anlnyse,wq_inst,nopt,x_romver,testcase):
    result=[]
    problem = {
        'num_vars':4,
        'names': [],
        'groups': None,
        'bounds': []}
    problem['num_vars']=len(snew)
    for i in range(len(snew)):
        k=i+1
        problem['names'].append(str("x")+str(k))
        if(snew[i]<=sw[i]):
            problem['bounds'].append([snew[i],sw[i]])
        else:
            problem['bounds'].append([sw[i], snew[i]])
    #x = sample(problem, N=int(count/(len(snew)+1)), num_levels=4, optimal_trajectories=None)
    x = sample(problem, N=5, num_levels=4, optimal_trajectories=None)
    #res_count=int(count/(len(snew)+1))*(len(snew)+1)
    res_count = 5* (len(snew) + 1)
    xf = np.zeros(res_count)
    for i in range(res_count):
        x_new = list(x[i, :])
        if (len(none_anlnyse) > 0):
            x_new = list(recover_demon(x_new, none_anlnyse,x_romver))
        xf[i] = EvalObjF(nopt, x_new, wq_inst, result, none_anlnyse,x_romver, testcase=testcase)
    #result=np.array(result)
    xf=np.array(xf)
    x=np.array(x)
    Si = morris.analyze(
        problem,
        x,
        xf,
        conf_level=0.95,
        print_to_console=True,
        num_levels=4,
        num_resamples=100,
    )
    si_path = os.getcwd() + os.sep + "Si_dymatic.txt"
    si_file = open(si_path, "w+")
    for i in range(len(Si["mu"])):
        si_file.write(str(Si["mu"][i]) + str('    ') + str(Si["mu_star"][i]) + '\n')
    si_file.write(str("-----------------------------------------")+'\n')
    si_file.close()
    new_add_anlnyse=[]
    new_add_romver=[]
    temp_none_anlnyse=[]
    temp_x=[]
    for i in range(len(snew)):
        temp_x.append(999+i)
    flag=True
    si_all=0.0
    for i in range(len(snew)):
        si_all+=float(Si["mu_star"][i])
    for i in range(len(snew)):
        if (float(Si["mu_star"][i])<= 0.1):
            new_add_anlnyse.append(i)
            if(flag):
                temp_x_new = recover_demon(temp_x, none_anlnyse, x_romver)
                flag=False
            #temp_x_new=recover_demon(temp_x,none_anlnyse,x_romver)
            x_romver.append(x[0][i])
            new_add_romver.append(x[0][i])
            insert=i
           # flag=True
            for k in range(len(temp_x_new)):
                if(temp_x[i]==temp_x_new[k]):
                    #for m in x_romver:
                       # if(m==temp_x[i]):
                           # flag=False
                          #  break
                    #if(flag):
                       # insert = k
                       # none_anlnyse.append(insert)
                    insert=k
                    none_anlnyse.append(insert)
    none_anlnyse_new=np.sort(np.array(none_anlnyse))
    x_romver_new=[]
    for i in range(len(none_anlnyse_new)):
        for j in range(len(none_anlnyse)):
            if (none_anlnyse_new[i] == none_anlnyse[j]):
                x_romver_new.append(x_romver[j])

    return x,xf,none_anlnyse_new,x_romver_new,new_add_anlnyse,new_add_romver
