import numpy as np 
import matplotlib.pyplot as plt

import matplotlib.font_manager as fmg

'''
for i, item in enumerate(fmg.findSystemFonts()):
    try:
        name=fmg.FontProperties(fname=item).get_name()
        if 'TIMES' in name.upper():
            print(name, item, i)
    except:
        pass
'''
para = "x"

font1={'family' : 'Times New Roman',
'weight' : 2,
'size' : 23,
}

all_font = fmg.findSystemFonts()
font = fmg.FontProperties(fname="C:/windows/fonts/times.ttf",size=23,weight=2)

if para == "pred":
    # result without adversarial attacking
    result_none_3 = 0.05278*np.ones(5)
    result_none_6 = 0.07463*np.ones(5)
    result_none_9 = 0.09011*np.ones(5)
    
    # result with gaussian white noise
    result_GWN_3 = np.array([0.05655,0.05592,0.05511,0.05425,0.05357])
    result_GWN_6 = np.array([0.07709,0.07669,0.07617,0.07565,0.07525])
    result_GWN_9 = np.array([0.092,0.09181,0.09126,0.09081,0.09065])
    
    # result with FGSM
    result_FGSM_3 = np.array([0.07712,0.07362,0.06977,0.06544,0.06037])
    result_FGSM_6 = np.array([0.09823,0.09472,0.09082,0.08637,0.08123])
    result_FGSM_9 = np.array([0.10979,0.10663,0.10318,0.09934,0.09497])
    
    # result with Target-FGSM
    result_TFGSM_3 = np.array([0.09339,0.08680,0.07983,0.07225,0.06380])
    result_TFGSM_6 = np.array([0.12705,0.11814,0.10882,0.09887,0.08814])
    result_TFGSM_9 = np.array([0.14941,0.13966,0.12938,0.11824,0.10628])
    
    # result with BIM
    result_iFGSM_3 = np.array([0.10421,0.09619,0.08544,0.07494,0.06472])
    result_iFGSM_6 = np.array([0.13017,0.12167,0.10828,0.09685,0.08580])
    result_iFGSM_9 = np.array([0.14023,0.13258,0.11942,0.10890,0.09901])
    
    # result with Target BIM
    result_iTFGSM_3 = np.array([0.11466,0.10402,0.09147,0.07929,0.06652])
    result_iTFGSM_6 = np.array([0.15674,0.14197,0.12478,0.10820,0.09155])
    result_iTFGSM_9 = np.array([0.18239,0.16616,0.14716,0.12891,0.11020])
    
    e = [0.01,0.008,0.006,0.004,0.002]
    
    # result with universal attacking
    result_universal_3 = np.array([0.07303,0.07105,0.06896,0.06602,0.06173])
    result_universal_6 = np.array([0.09215,0.09043,0.08873,0.08638,0.08273])
    result_universal_9 = np.array([0.10443,0.10307,0.10174,0.09990,0.09678])

    # result with universal attacking
    result_universal_3_new = np.array([0.23309,0.19995,0.15769,0.11634,0.07970])
    result_universal_6_new = np.array([0.31972,0.28488,0.22138,0.15654,0.10267])
    result_universal_9_new = np.array([0.36692,0.33334,0.26005,0.17928,0.11769])

    plt.figure(figsize=(16,8))
    
    plt.plot(e,result_none_3,color='#fc5a50',linewidth=5,label="None",marker="o",markersize=24)
    plt.plot(e,result_GWN_3,color='#f9bc08',linewidth=5,label="GWN",marker="v",markersize=24)
    plt.plot(e,result_FGSM_3,color='#90b134',linewidth=5,label="FGSM",marker="s",markersize=24)
    plt.plot(e,result_TFGSM_3,color='#0cdc73',linewidth=5,label="TFGSM",marker="h",markersize=24)
    plt.plot(e,result_iFGSM_3,color='#448ee4',linewidth=5,label="BIM",marker="d",markersize=24)
    plt.plot(e,result_iTFGSM_3,color='#e03fd8',linewidth=5,label="TBIM",marker="*",markersize=24)
    plt.plot(e,result_universal_3_new,color='#6140ef',linewidth=5,label="UNI",marker="^",markersize=24)

    plt.xlabel("Energy Proportion",FontProperties=font)
    plt.ylabel("MAPE",FontProperties=font)
    plt.title("15min prediction",FontProperties=font)
    plt.legend(loc="upper left",prop=font1)
    plt.ylim(0.05,0.24)
    plt.xticks(e)
    plt.tick_params(labelsize=23,direction="in",width=3,length=9)
    
    bwith = 3
    ax = plt.gca()#获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    
    plt.show()

else:
    x_test = np.load("./output/ypred/xtest.npy")
    x_perturbed = np.load('./output/ypred/xtest_universal_002.npy')
    x_tbim = np.load('./output/ypred/xtest_tbim.npy')
    y_test = np.load('./output/ypred/ytest_none.npy')
    y_perturbed = np.load('./output/ypred/ytest_universal_002.npy')
    y_tbim = np.load('./output/ypred/ytest_tbim.npy')


    pp = 190
    qq = 190

    plt.figure(figsize=(16,8))
    index = np.linspace(0,np.size(y_test,1),num=np.size(y_test,1))
    y1 = y_test[1,:,pp,0]
    y2 = y_perturbed[1,:,pp,0]
    y3 = y_tbim[1,:,pp,0]
    x1 = x_test[:,11,qq,0]
    x2 = x_perturbed[:,11,qq,0]
    x3 = x_tbim[:,11,qq,0]
#    plt.figure(figsize=(16,8))
    
    plt.plot(index,y1,color='#448ee4',ls='-',linewidth=2,label="None")
    plt.plot(index,y2,color='#e03fd8',ls='--',linewidth=2,label="0.2%-UNI")
    plt.plot(index,y3,color='#2fef10',ls='-.',linewidth=2,label="0.7%-TBIM")
    
#    plt.plot(index,x1,color='#448ee4',ls='-',linewidth=2,label="None")
#    plt.plot(index,x2,color='#e03fd8',ls='--',linewidth=2,label="0.2%-UNI")

    plt.xlabel("Time Step",FontProperties=font)
    plt.ylabel("Traffic",FontProperties=font)
    plt.title("30min prediction",FontProperties=font)
#    plt.title("Input Comparison",FontProperties=font)
    plt.legend(loc="upper left",prop=font1)
#    plt.ylim(0.05,0.12)
#    plt.xticks(e)
    plt.tick_params(labelsize=23,direction="in",width=3,length=9)
    
    bwith = 3
    ax = plt.gca()#获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    
    plt.show()
