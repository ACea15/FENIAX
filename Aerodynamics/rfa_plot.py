import numpy as np
import matplotlib.pyplot as plt
import pdb
#import rfa
import Aerodynamics.rfa as rfa
###########################################################################################################################################
#Plot results
###########################################################################################################################################
def plot1(reduced_freq,plot_points,poles,RFA_mat,RFA_Method,aero_matrices_real,aero_matrices_imag):
        font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }

        box = dict(facecolor='yellow', pad=5, alpha=0.2)
        fig = plt.figure(figsize=(13.,4.))
        #fig = plt.figure()
        fig.subplots_adjust(left=0.,right=1., wspace=0.)

        num_plot_points=len(plot_points[:,0])
        num_reduced_freq = len(reduced_freq)
        k_max = reduced_freq[-1]
        k=0
        step=1000
        AIC_Roger_Real_print=np.zeros((num_plot_points,step))
        AIC_Roger_Imag_print=np.zeros((num_plot_points,step))
        AIC_input_Real_print=np.zeros((num_plot_points,num_reduced_freq))
        AIC_input_Imag_print=np.zeros((num_plot_points,num_reduced_freq))
        len_reduced_freq=np.zeros((num_plot_points))

        for r in range (num_plot_points):

            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            RB_Modes_QS = ''
            if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
                delta_freq=k_max_RB/step
                num_reduced_freq_=num_reduced_freq_QS
            else:
                delta_freq=k_max/step
                num_reduced_freq_=num_reduced_freq
            len_reduced_freq[r]=num_reduced_freq_

            AIC_Roger_Real=np.zeros((step))
            AIC_Roger_Imag=np.zeros((step))

            Freq_plot=np.zeros((step))
            k_red=0
            for i in range (step):
                Freq_plot[i]=k_red
                k_red=k_red+delta_freq

            k=k+1
            p=int(plot_points[r,0]-1)
            q=int(plot_points[r,1]-1)
            AIC_input_Imag=np.zeros((num_reduced_freq_))
            AIC_input_Real=np.zeros((num_reduced_freq_))

            for red_freq in range(num_reduced_freq):
                AIC_input_Real[red_freq]=aero_matrices_real[red_freq][p,q]
                AIC_input_Imag[red_freq]=aero_matrices_imag[red_freq][p,q]


            k_red=0
            for i in range (step):
                # AIC_Roger_Real[i]=RFA_mat[0][p,q]-(k_red**2)*RFA_mat[2][p,q]
                # AIC_Roger_Imag[i]=(k_red)*RFA_mat[1][p,q]
                # if RFA_Method == 'E' or RFA_Method == 'e':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((poles[pole])/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((-k_red)/((k_red**2)+poles[pole]**2))
                # if RFA_Method == 'R' or RFA_Method == 'r':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((k_red**2)/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((k_red*poles[pole])/((k_red**2)+poles[pole]**2))
                AIC = rfa.RFA_freq(poles,k_red,RFA_mat,RFA_Method)
                AIC_Roger_Real[i] = AIC.real[p,q]
                AIC_Roger_Imag[i] = AIC.imag[p,q]

                k_red=k_red+delta_freq

            if num_plot_points == 1 :
                sub_plt= fig.add_subplot(1,1,k)
            if num_plot_points == 2 :
                sub_plt= fig.add_subplot(1,2,k)
            if num_plot_points == 3 :
                sub_plt= fig.add_subplot(1,3,k)
            if num_plot_points == 4 :
                sub_plt= fig.add_subplot(2,2,k)
            if num_plot_points == 5 or num_plot_points == 6 :
                sub_plt= fig.add_subplot(2,3,k)
            if num_plot_points >= 7 and num_plot_points <= 9 :
                sub_plt= fig.add_subplot(3,3,k)
            if num_plot_points >= 10 and num_plot_points <= 12 :
                sub_plt= fig.add_subplot(3,4,k)
            if num_plot_points >= 13 and num_plot_points <= 16 :
                sub_plt= fig.add_subplot(4,4,k)

            sub_plt.plot(AIC_Roger_Real,AIC_Roger_Imag,c='steelblue',label='RFA')
            sub_plt.plot(AIC_input_Real,AIC_input_Imag,'ok',mfc='none',markersize=3,label='DLM')
            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            # if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
            #     for i, txt in enumerate(reduced_freq_QS):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            # else:
            #     for i, txt in enumerate(reduced_freq):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            sub_plt.set_title('AIC ['+str(int(plot_points[r,0])+1)+','+str(int(plot_points[r,1])+1)+']', fontdict=font)
            #sub_plt.xaxis.set_label_coords(-0.3, 0.5)
            #sub_plt.yaxis.set_label_coords(-0.3, 0.5)
            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),x=-0.1,y=1.)
            if r ==0:
                sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)
                #sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            elif r==3:
                sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)
            elif r==2:
                sub_plt.legend(bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)
            elif r==6:
                sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)
                sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            elif r==7:
                 sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            elif r==8:
                 sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            #sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            #sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)

            AIC_Roger_Real_print[r,:]=AIC_Roger_Real[:]
            AIC_Roger_Imag_print[r,:]=AIC_Roger_Imag[:]
            for q in range (num_reduced_freq_):
                AIC_input_Real_print[r,q]=AIC_input_Real[q]
                AIC_input_Imag_print[r,q]=AIC_input_Imag[q]

        #plt.savefig('AIC_Plot.png')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        #plt.savefig('gaf2.pdf',dpi=300, bbox_inches='tight')
        #plt.close(fig)
        plt.show()

def plot3(reduced_freq,plot_points,poles,poles2,RFA_mat,RFA_mat2,RFA_Method,aero_matrices_real,aero_matrices_imag,save=0,save_name='AIC.pdf'):
        font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 11,
        }

        box = dict(facecolor='yellow', pad=5, alpha=0.2)
        fig = plt.figure(figsize=(8.,5.))
        #plt.rc('text', usetex=True)
        #fig = plt.figure()
        fig.subplots_adjust(left=0., wspace=0.)

        num_plot_points=len(plot_points[:,0])
        num_reduced_freq = len(reduced_freq)
        k_max = reduced_freq[-1]
        k=0
        step=1000
        AIC_Roger_Real_print=np.zeros((num_plot_points,step))
        AIC_Roger_Imag_print=np.zeros((num_plot_points,step))
        AIC_input_Real_print=np.zeros((num_plot_points,num_reduced_freq))
        AIC_input_Imag_print=np.zeros((num_plot_points,num_reduced_freq))
        len_reduced_freq=np.zeros((num_plot_points))

        for r in range (num_plot_points):

            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            RB_Modes_QS = ''
            if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
                delta_freq=k_max_RB/step
                num_reduced_freq_=num_reduced_freq_QS
            else:
                delta_freq=k_max/step
                num_reduced_freq_=num_reduced_freq
            len_reduced_freq[r]=num_reduced_freq_

            # AIC_Roger_Real=np.zeros((step))
            # AIC_Roger_Imag=np.zeros((step))
            AIC_Roger_Real=np.zeros((step))
            AIC_Roger_Imag=np.zeros((step))
            AIC_Roger_Real2=np.zeros((step))
            AIC_Roger_Imag2=np.zeros((step))
            Freq_plot=np.zeros((step))
            k_red=0
            for i in range (step):
                Freq_plot[i]=k_red
                k_red=k_red+delta_freq

            k=k+1
            p=int(plot_points[r,0]-1)
            q=int(plot_points[r,1]-1)
            AIC_input_Imag=np.zeros((num_reduced_freq_))
            AIC_input_Real=np.zeros((num_reduced_freq_))

            for red_freq in range(num_reduced_freq):
                AIC_input_Real[red_freq]=aero_matrices_real[red_freq][p,q]
                AIC_input_Imag[red_freq]=aero_matrices_imag[red_freq][p,q]


            k_red=0
            for i in range (step):
                # AIC_Roger_Real[i]=RFA_mat[0][p,q]-(k_red**2)*RFA_mat[2][p,q]
                # AIC_Roger_Imag[i]=(k_red)*RFA_mat[1][p,q]
                # if RFA_Method == 'E' or RFA_Method == 'e':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((poles[pole])/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((-k_red)/((k_red**2)+poles[pole]**2))
                # if RFA_Method == 'R' or RFA_Method == 'r':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((k_red**2)/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((k_red*poles[pole])/((k_red**2)+poles[pole]**2))
                # AIC = rfa.RFA_freq(poles,k_red,RFA_mat,RFA_Method)
                # AIC_Roger_Real[i] = AIC.real[p,q]
                # AIC_Roger_Imag[i] = AIC.imag[p,q]
                AIC = rfa.RFA_freq(poles,k_red,RFA_mat,RFA_Method)
                AIC_Roger_Real[i] = AIC.real[p,q]
                AIC_Roger_Imag[i] = AIC.imag[p,q]
                AIC2 = rfa.RFA_freq(poles2,k_red,RFA_mat2,RFA_Method)
                AIC_Roger_Real2[i] = AIC2.real[p,q]
                AIC_Roger_Imag2[i] = AIC2.imag[p,q]

                k_red=k_red+delta_freq

            if num_plot_points == 1 :
                sub_plt= fig.add_subplot(1,1,k)
            if num_plot_points == 2 :
                sub_plt= fig.add_subplot(1,2,k)
            if num_plot_points == 3 :
                sub_plt= fig.add_subplot(1,3,k)
            if num_plot_points == 4 :
                sub_plt= fig.add_subplot(2,2,k)
            if num_plot_points == 5 or num_plot_points == 6 :
                sub_plt= fig.add_subplot(2,3,k)
            if num_plot_points >= 7 and num_plot_points <= 9 :
                sub_plt= fig.add_subplot(3,3,k)
            if num_plot_points >= 10 and num_plot_points <= 12 :
                sub_plt= fig.add_subplot(3,4,k)
            if num_plot_points >= 13 and num_plot_points <= 16 :
                sub_plt= fig.add_subplot(4,4,k)

            sub_plt.plot(AIC_Roger_Real,AIC_Roger_Imag,c='steelblue',label='RFA')
            sub_plt.plot(AIC_Roger_Real2,AIC_Roger_Imag2,c='lightsalmon',label='RFA Opt')
            sub_plt.plot(AIC_input_Real[0::3],AIC_input_Imag[0::3],'ok',mfc='none',markersize=3,label='DLM')
            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            # if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
            #     for i, txt in enumerate(reduced_freq_QS):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            # else:
            #     for i, txt in enumerate(reduced_freq):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            #sub_plt.set_title(r'AIC_{%s%s}'%(str(int(plot_points[r,0])),str(int(plot_points[r,1]))))
            sub_plt.set_title(r'$AIC_{%s%s}$'%(str(int(plot_points[r,0])+1),str(int(plot_points[r,1])+1)))
            #sub_plt.xaxis.set_label_coords(-0.3, 0.5)
            #sub_plt.yaxis.set_label_coords(-0.3, 0.5)
            #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),x=-0.1,y=1.)
            if r ==0:
                sub_plt.set_ylabel('Imag', fontsize=11,x=-0.1,y=0.5, fontdict=font)
                #sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            elif r==3:
                sub_plt.set_ylabel('Imag', fontsize=11,x=-0.1,y=0.5, fontdict=font)
            elif r==2:
                sub_plt.legend(bbox_to_anchor=(1.05, 1.05), loc=2, borderaxespad=0.)
            elif r==6:
                sub_plt.set_ylabel('Imag', fontsize=11,x=-0.1,y=0.5, fontdict=font)
                sub_plt.set_xlabel('Real', fontsize=11,x=0.5,y=-0.1, fontdict=font,)
            elif r==7:
                 sub_plt.set_xlabel('Real', fontsize=11,x=0.5,y=-0.1, fontdict=font,)
            elif r==8:
                 sub_plt.set_xlabel('Real', fontsize=11,x=0.5,y=-0.1, fontdict=font,)
            #sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            #sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)

            AIC_Roger_Real_print[r,:]=AIC_Roger_Real[:]
            AIC_Roger_Imag_print[r,:]=AIC_Roger_Imag[:]
            for q in range (num_reduced_freq_):
                AIC_input_Real_print[r,q]=AIC_input_Real[q]
                AIC_input_Imag_print[r,q]=AIC_input_Imag[q]

        #plt.savefig('AIC_Plot.png')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        if save:
           plt.savefig(save_name,dpi=300, bbox_inches='tight')
           plt.close(fig)
        plt.show()


def plot2(reduced_freq,plot_points,poles,poles2,RFA_mat,RFA_mat2,RFA_Method,aero_matrices_real,aero_matrices_imag):
        font = {'family' : 'serif',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 10,
        }

        box = dict(facecolor='yellow', pad=5, alpha=0.2)
        fig = plt.figure(figsize=(8.,7.))
        fig.subplots_adjust(left=0.2, wspace=0.6)

        num_plot_points=len(plot_points[:,0])
        num_reduced_freq = len(reduced_freq)
        k_max = reduced_freq[-1]
        k=0
        step=1000
        AIC_Roger_Real_print=np.zeros((num_plot_points,step))
        AIC_Roger_Imag_print=np.zeros((num_plot_points,step))
        AIC_input_Real_print=np.zeros((num_plot_points,num_reduced_freq))
        AIC_input_Imag_print=np.zeros((num_plot_points,num_reduced_freq))
        len_reduced_freq=np.zeros((num_plot_points))

        for r in range (num_plot_points):

            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            RB_Modes_QS = ''
            if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
                delta_freq=k_max_RB/step
                num_reduced_freq_=num_reduced_freq_QS
            else:
                delta_freq=k_max/step
                num_reduced_freq_=num_reduced_freq
            len_reduced_freq[r]=num_reduced_freq_

            AIC_Roger_Real=np.zeros((step))
            AIC_Roger_Imag=np.zeros((step))
            AIC_Roger_Real2=np.zeros((step))
            AIC_Roger_Imag2=np.zeros((step))
            Freq_plot=np.zeros((step))
            k_red=0
            for i in range (step):
                Freq_plot[i]=k_red
                k_red=k_red+delta_freq

            k=k+1
            p=int(plot_points[r,0]-1)
            q=int(plot_points[r,1]-1)
            AIC_input_Imag=np.zeros((num_reduced_freq_))
            AIC_input_Real=np.zeros((num_reduced_freq_))

            for red_freq in range(num_reduced_freq):
                AIC_input_Real[red_freq]=aero_matrices_real[red_freq][p,q]
                AIC_input_Imag[red_freq]=aero_matrices_imag[red_freq][p,q]


            k_red=0
            for i in range (step):
                # AIC_Roger_Real[i]=RFA_mat[0][p,q]-(k_red**2)*RFA_mat[2][p,q]
                # AIC_Roger_Imag[i]=(k_red)*RFA_mat[1][p,q]
                # if RFA_Method == 'E' or RFA_Method == 'e':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((poles[pole])/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((-k_red)/((k_red**2)+poles[pole]**2))
                # if RFA_Method == 'R' or RFA_Method == 'r':
                #     for pole in range(npoles):
                #         AIC_Roger_Real[i]=AIC_Roger_Real[i]+RFA_mat[3+pole][p,q]*((k_red**2)/((k_red**2)+poles[pole]**2))
                #         AIC_Roger_Imag[i]=AIC_Roger_Imag[i]+RFA_mat[3+pole][p,q]*((k_red*poles[pole])/((k_red**2)+poles[pole]**2))
                AIC = rfa.RFA_freq(poles,k_red,RFA_mat,RFA_Method)
                AIC_Roger_Real[i] = AIC.real[p,q]
                AIC_Roger_Imag[i] = AIC.imag[p,q]
                AIC2 = rfa.RFA_freq(poles2,k_red,RFA_mat2,RFA_Method)
                AIC_Roger_Real2[i] = AIC2.real[p,q]
                AIC_Roger_Imag2[i] = AIC2.imag[p,q]

                k_red=k_red+delta_freq

            if num_plot_points == 1 :
                sub_plt= fig.add_subplot(1,1,k)
            if num_plot_points == 2 :
                sub_plt= fig.add_subplot(1,2,k)
            if num_plot_points == 3 :
                sub_plt= fig.add_subplot(1,3,k)
            if num_plot_points == 4 :
                sub_plt= fig.add_subplot(2,2,k)
            if num_plot_points == 5 or num_plot_points == 6 :
                sub_plt= fig.add_subplot(2,3,k)
            if num_plot_points >= 7 and num_plot_points <= 9 :
                sub_plt= fig.add_subplot(3,3,k)
            if num_plot_points >= 10 and num_plot_points <= 12 :
                sub_plt= fig.add_subplot(3,4,k)
            if num_plot_points >= 13 and num_plot_points <= 16 :
                sub_plt= fig.add_subplot(4,4,k)

            sub_plt.plot(AIC_Roger_Real,AIC_Roger_Imag,c='steelblue',label='RFA')
            sub_plt.plot(AIC_Roger_Real2,AIC_Roger_Imag2,c='lightgreen',label='RFA Opt')
            sub_plt.plot(AIC_input_Real,AIC_input_Imag,'k.--',label='DLM')
            #if RB_Modes_QS == 'y' and (plot_points[r,1]<=6) and (Matrix_type=='qaa'):
            # if RB_Modes_QS == 'y' and (plot_points[r,1]<=6):
            #     for i, txt in enumerate(reduced_freq_QS):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            # else:
            #     for i, txt in enumerate(reduced_freq):
            #         sub_plt.annotate( txt , (AIC_input_Real[i],AIC_input_Imag[i]))
            sub_plt.set_title('AIC ['+str(int(plot_points[r,0]))+','+str(int(plot_points[r,1]))+']', fontdict=font)
            sub_plt.xaxis.set_label_coords(-0.3, 0.5)
            sub_plt.yaxis.set_label_coords(-0.3, 0.5)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0),x=-0.1,y=1.)
            sub_plt.set_xlabel('Real', fontsize=9,x=0.5,y=-0.1, fontdict=font,)
            sub_plt.set_ylabel('Imag', fontsize=9,x=-0.1,y=0.5, fontdict=font)

            AIC_Roger_Real_print[r,:]=AIC_Roger_Real[:]
            AIC_Roger_Imag_print[r,:]=AIC_Roger_Imag[:]
            for q in range (num_reduced_freq_):
                AIC_input_Real_print[r,q]=AIC_input_Real[q]
                AIC_input_Imag_print[r,q]=AIC_input_Imag[q]

        #plt.savefig('AIC_Plot.png')

        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend()
        #plt.savefig('gaf2.pdf',dpi=300, bbox_inches='tight')
        #plt.close(fig)
        plt.show()
