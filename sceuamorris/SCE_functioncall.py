#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version -  helpfunctions
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S. 2011
# Purpose:
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
import logging
import math
import os
import sys
import numpy as np
import time

from SALib.sample.morris import sample


##print sys.path[0]

##sys.path.append('D:\Modellen\Snippets_for_all\SCE')
################################################################################
## Modrun  is the user part of the script:
## gets number of parameters and an array x with these parameters
## let your function or model run/evaluate and give the evaluation value back
##------------------------------------------------------------------------------

##def Modrun(npar,x):
##	'''
##	User has to put in his model to run
##	'''
##
##    return f

################################################################################


################################################################################
##  Sampling called from SCE
################################################################################
def mse(evaluation, simulation):
    """
    Mean Squared Error

        .. math::

         MSE=\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Mean Squared Error
    :rtype: float
    """

    if len(evaluation) == len(simulation):
        obs, sim = np.array(evaluation), np.array(simulation)
        mse = np.nanmean((obs - sim) ** 2)
        return mse
    else:
        logging.warning(
            "evaluation and simulation lists does not have the same length."
        )
        return np.nan
def rmse(evaluation, simulation):
    """
    Root Mean Squared Error

        .. math::

         RMSE=\\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(e_{i}-s_{i})^2}

    :evaluation: Observed data to compared with simulation data.
    :type: list

    :simulation: simulation data to compared with evaluation data
    :type: list

    :return: Root Mean Squared Error
    :rtype: float
    """
    if len(evaluation) == len(simulation) > 0:
        return np.sqrt(mse(evaluation, simulation))
    else:
        logging.warning("evaluation and simulation lists do not have the same length.")
        return np.nan

def SampleInputMatrix(nrows,npars,bu,bl,iseed,distname='randomUniform'):
    '''
    Create inputparameter matrix for nrows simualtions,
    for npars with bounds ub and lb (np.array from same size)
    distname gives the initial sampling ditribution (currently one for all parameters)

    returns np.array
    '''
    np.random.seed(iseed)
    x=np.zeros((nrows,npars))
    bound = bu-bl
    #problem={
       # 'num_vars': 4,
     #   'names': ['x1', 'x2', 'x3', 'x4'],
     #   'groups': None,
     #   'bounds':[]
   # }
   # for i in range(npars):
   #     if(Si["mu"][i]==0):
    #        bu[i]=bl[i]+0.1
   #     problem["bounds"].append([bl[i],bu[i]])
   # if(nrows<=(npars+1)):
   #     x = sample(problem, N=1, num_levels=4, optimal_trajectories=None)
   # else:
   #     x=sample(problem, N=(int(nrows/(npars+1))), num_levels=4, optimal_trajectories=None)
    for i in range(nrows):
##        x[i,:]= bl + DistSelector([0.0,1.0,npars],distname='randomUniform')*bound  #only used in full Vhoeys-framework
        x[i,:]= bl + np.random.rand(1,npars)*bound
    return x


################################################################################
##    TESTFUNCTIONS TO CHECK THE IMPLEMENTATION OF THE SCE ALGORITHM
################################################################################
def Ackley(x):
    res11 = 0
    res12 = 0
    for i in range(len(x)):
        res11 += x[i] ** 2
        res12 += math.cos(2 * math.pi * x[i])
    return (-20)*math.exp((-0.2)*math.sqrt(res11/len(x)))-math.exp(res12/len(x))+20+math.e
def Rastrigin(x):
    res3 = 0
    for i in range(len(x)):
        res3 += x[i] ** 2 - 10 * math.cos(2 * math.pi * x[i]) + 10
    return res3
def Weierstrass(x):
    res51 = 0
    res52 = 0
    for i in range(len(x)):
        for k in range(21):
            res51 += ((0.5) ** k) * math.cos(2 * math.pi * (3 ** k) * (x[i] + 0.5))
            res52 += ((0.5) ** k) * math.cos(2 * math.pi * (3 ** k) * 0.5)
    res5 = res51 - len(x) * res52
    return res5
def Griewank(x):
    res71 = 0
    res72 = 1
    for i in range(len(x)):
        res71 += (x[i] ** 2) / 4000
    for i in range(len(x)):
        res72 *= math.cos(x[i] / math.sqrt(i + 1))
    res7 = res71 - res72 + 1
    return res7
def Sphere(x):
    res9 = 0
    for i in range(len(x)):
        res9 += x[i] ** 2
    return res9
"""CF6"""
def testfunctn1(nopt,x,wq_inst,Y):
    filepath=os.getcwd()
    para_path=filepath+os.sep+"param_40.txt"
    if(not os.path.exists(para_path)):
        params_file=open(para_path,"w+")
        for i in x:
            params_file.write(str(i)+'  ')
        params_file.write('\n')
        params_file.close()
    else:
        params_file = open(para_path, "r+")
        params_list=params_file.readlines()
        params_file.close()
        params_file = open(para_path, "w+")
        for i in params_list:
            params_file.write(i)
        for i in x:
            params_file.write(str(i) + '  ')
        params_file.write('\n')
        params_file.close()
    res=Ackley(x)
    like = res
    like_path = filepath + os.sep + 'like_40.txt'
    if(not os.path.exists(like_path)):
        like_para = open(like_path, 'w+')
        like_para.write(str(like) + '\n')
        like_para.close()
    else:
        like_para = open(like_path, 'r+')
        like_list = like_para.readlines()
        like_para.close()
        like_para = open(like_path, 'w+')
        count = 0
        for i in like_list:
            like_para.write(i)
            count += 1
        like_para.write(str(like) + '\n')
        print("第%d次\n" % (count + 1))
        like_para.close()
    # Y.append(like)
    wq_res = 0
    return like



def testfunctn6(nopt,x,wq_inst,Y):
    '''
        This is the Goldstein-Price Function
        Bound X1=[-2,2], X2=[-2,2]
        Global Optimum: 3.0,(0.0,-1.0)
        '''
    #wq_inst = [0.0143, 1.11325, 0.025, 10.094]
    # 8个参数的值
    params_values = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]]
    # 获取当前工作路径
    filepath = os.getcwd()
    # 输出每组数据
    params_path = filepath + os.sep + "params.txt"
    if (not os.path.exists(params_path)):
        params = open(params_path, "w+")
        for i in params_values:
            params.write(str(i) + '  ')
        params.write('\n')
        params.close()
    else:
        params = open(params_path, "r+")
        params_list = params.readlines()
        params = open(params_path, "w+")
        for i in params_list:
            params.write(i)
        for i in params_values:
            params.write(str(i) + '  ')
        params.write('\n')
        params.close()
    # 需要修改的输入文件名
    wq3djnp_path = filepath + os.sep + "wq_3dwc.jnp"
    # 修改文件
    wq3djnp = open(wq3djnp_path, "r+")
    wqlist = wq3djnp.readlines()
    wq3djnp.close()
    # 修改第一个参数
    wqlist[76] = str(wqlist[76])[:27] + ' ' + str(params_values[0]) + ',' + '\n'
    # 修改第二个参数
    wqlist[139] = str(wqlist[139])[:15] + ' ' + str(params_values[1]) + ',' + '\n'
    # 修改第三个参数
    wqlist[143] = str(wqlist[143])[:15] + ' ' + str(params_values[2]) + ',' + '\n'
    # 修改第四个参数
    wqlist[149] = str(wqlist[149])[:15] + ' ' + str(params_values[3]) + '\n'
    # 修改第五个参数
    wqlist[152] = str(wqlist[152])[:15] + ' ' + str(params_values[4]) + '\n'
    # 修改第六个参数
    wqlist[159] = str(wqlist[159])[:29] + ' ' + str(params_values[5]) + ',' + '\n'
    # 修改第七个参数
    wqlist[160] = str(wqlist[160])[:30] + ' ' + str(params_values[6]) + ',' + '\n'
    # 修改第八个参数
    wqlist[158] = str(wqlist[158])[:16] + ' ' + str(params_values[7]) + ',' + '\n'
    wq3djnp = open(wq3djnp_path, "w+")
    for i in wqlist:
        wq3djnp.write(i)
    wq3djnp.close()
    wb3djnp_path = filepath + os.sep + "wq_biota.jnp"
    wb3djnp = open(wb3djnp_path, "r+")
    wblist = wb3djnp.readlines()
    wb3djnp.close()
    # 修改wbjnp文件
    # 修改wb第一个参数
    wblist[44] = str(wblist[44])[:26] + ' ' + str(round(params_values[8], 3)) + ',' + '\n'
    # 修改wb第二个参数
    wblist[45] = str(wblist[45])[:33] + ' ' + str(round(params_values[9], 3)) + ',' + '\n'
    # 修改wb第二个参数
    wblist[46] = str(wblist[46])[:33] + ' ' + str(round(params_values[10], 3)) + ',' + '\n'
    wb3djnp = open(wb3djnp_path, "w+")
    for i in wblist:
        wb3djnp.write(i)
    wb3djnp.close()
    # 运行文件生成模拟结果
    main = "EFDCPlus_MPI_11.6.exe"

    result = os.system(
        "start cmd.exe /k mpiexec -n 8  F:\\morris与sceua合并试验\\SCE-UA-EFDC\\EFDCPlus_MPI_11.6.exe")
    print(result)
    wq_daily = []
    total_day = 10
    output = filepath + os.sep + "#output"
    count = 1
    while not os.path.exists(output + os.sep + "WQ_WCRST_20220707_00.OUT"):
        time.sleep(30)
    os.system("taskkill /F /IM EFDCPlus_MPI_11.6.exe")
    os.system("taskkill /F /IM cmd.exe")
    if result == 0:
        dop = []
        don = []
        nhx = []
        do = []
        # 获取运行结果文件
        while count <= total_day:
            dop_all = 0
            don_all = 0
            nhx_all = 0
            do_all = 0
            if (6 + count) < 10:
                filename = "WQ_WCRST_2022070" + str(int(6 + count)) + "_00.OUT"
            else:
                filename = "WQ_WCRST_202207" + str(int(6 + count)) + "_00.OUT"
            wq_result_daily = open(output + os.sep + filename, "r+")
            wq_result_daily_list = wq_result_daily.readlines()
            for i in range(4, 2416):
                # print(int(str(wq_result_daily_list[i])[107:109]))
                dop.append(
                    float(str(wq_result_daily_list[i])[82:90]) * pow(10, int(str(wq_result_daily_list[i])[91:95])))
                don.append(
                    float(str(wq_result_daily_list[i])[138:146]) * pow(10, int(str(wq_result_daily_list[i])[147:151])))
                nhx.append(
                    float(str(wq_result_daily_list[i])[152:160]) * pow(10, int(str(wq_result_daily_list[i])[161:165])))
                do.append(
                    float(str(wq_result_daily_list[i])[222:230]) * pow(10, int(str(wq_result_daily_list[i])[231:235])))
            wq_result_daily.close()
            for i in range(0, 2142, 3):
                dop_all += dop[i]
                don_all += don[i]
                nhx_all += nhx[i]
                do_all += do[i]
            dop_aver = dop_all / 804
            don_aver = don_all / 804
            nhx_aver = nhx_all / 804
            do_aver = do_all / 804
            wq_daily.extend([dop_aver, don_aver, nhx_aver, do_aver])
            dop.clear()
            don.clear()
            nhx.clear()
            do.clear()
            count += 1
        os.remove(output + os.sep + "WQ_WCRST_20220707_00.OUT")
    else:
        print("运行失败，查看原因************************************************")

    like = rmse(wq_inst, wq_daily)
    like_path = filepath + os.sep + 'like.txt'
    if (not os.path.exists(like_path)):
        like_para = open(like_path, 'w+')
        like_para.write(str(like) + '\n')
        like_para.close()
    else:
        like_para = open(like_path, 'r+')
        like_list = like_para.readlines()
        like_para = open(like_path, 'w+')
        count = 0
        for i in like_list:
            like_para.write(i)
            count += 1
        like_para.write(str(like) + '\n')
        print("第%d次\n" % (count + 1))
        like_para.close()
    # Y.append(like)
    wq_res = 0
    return like

def testfunctn2(nopt,x):
    '''
    %  This is the Rosenbrock Function
    %  Bound: X1=[-5,5], X2=[-2,8]; Global Optimum: 0,(1,1)
        bl=[-5 -5]; bu=[5 5]; x0=[1 1];
    '''

    x1 = x[0]
    x2 = x[1]
    a = 100.0
    f = a * (x2 - x1**2)**2 + (1 - x1)**2
    return f

def testfunctn3(nopt,x,Y):
    '''3
    %  This is the Six-hump Camelback Function.
    %  Bound: X1=[-5,5], X2=[-5,5]
    %  True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    '''
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 =x[4]
    x6=x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    #f = (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x3 ** 2 + x1 * x4 + (-4 + 4 * x2 ** 2) * x2 ** 2
    f = x2 ** 2 + x2 ** 2 + x3**2 - np.cos(18.0 * x2) - np.cos(18.0 * x2)+x5*x7+x6**2+2*x8*x9
    Y.append(f)
    return abs(f)

def testfunctn4(nopt,x):
    '''4
    %  This is the Rastrigin Function
    %  Bound: X1=[-1,1], X2=[-1,1]
    %  Global Optimum: -2, (0,0)
    '''
    x1 = x[0]
    x2 = x[1]
    f = x1**2 + x2**2 - np.cos(18.0*x1) - np.cos(18.0*x2)
    return f

def testfunctn5(nopt,x):
    '''
    This is the Griewank Function (2-D or 10-D)
    Bound: X(i)=[-600,600], for i=1,2,...,10
    Global Optimum: 0, at origin
    '''
    if nopt==2:
        d = 200.0
    else:
        d = 4000.0

    u1 = 0.0
    u2 = 1.0
    for j in range(nopt):
        u1 = u1 + x[j]**2/d
        u2 = u2 * np.cos(x[j]/np.sqrt(float(j+1)))

    f = u1 - u2 + 1
    return f


################################################################################
##   FUNCTION CALL FROM SCE-ALGORITHM !!
################################################################################

# def EvalObjF(npar,x,testcase=True,testnr=1):
def EvalObjF(npar, x,wq_inst,Y,none_anlnyse,x_romver,testcase=True):
    '''
    The SCE algorithm calls this function which calls the model itself
    (minimalisation of function output or evaluation criterium coming from model)
    and returns the evaluation function to the SCE-algorithm

    If testcase =True, one of the example tests are run
    '''
    #将不具备敏感性参数的数据放入原数组进行计算
    if(len(none_anlnyse)>0):
        x=list(x)
        for i in range(len(none_anlnyse)):
            rest=list(x[:none_anlnyse[i]])
            rest.append(x_romver[i])
            rest.extend(x[none_anlnyse[i]:])
            x=rest
        x=np.array(x)
    filepath=os.getcwd()
    if(not os.path.exists(filepath+os.sep+"param_1.txt")):
        param = open(filepath + os.sep + "param_1.txt","w+")
        param.write(str(list(x))+'\n')
        param.close()
    else:
        param = open(filepath + os.sep + "param_1.txt","r+")
        param_list=param.readlines()
        param.close()
        param = open(filepath + os.sep + "param_1.txt","w+")
        for i in param_list:
            param.write(i)
        param.write(str(list(x))+'\n')
        param.close()

##    print 'testnummer is %d' %testnr

    if testcase==True:
        res=testfunctn1(npar, x,wq_inst,Y)
        if(not os.path.exists(filepath+os.sep+"Y.txt")):
            Ytxt=open(filepath+os.sep+"Y.txt","w+")
            Ytxt.write(str(res)+'\n')
            Ytxt.close()
        else:
            Ytxt = open(filepath + os.sep + "Y.txt", "r+")
            Ytxt_list=Ytxt.readlines()
            Ytxt.close()
            Ytxt = open(filepath + os.sep + "Y.txt", "w+")
            for i in Ytxt_list:
                Ytxt.write(i)
            Ytxt.write(str(res)+'\n')
            Ytxt.close()
        return res
        #return testfunctn(npar,x,wq_inst,Y)
    else:
        # Welk model/welke objfunctie/welke periode/.... users keuze!
        return



