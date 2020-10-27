/*********************************************************************
 * model.cpp
 * Keep in mind:
 * <> Use 0-based indexing as always in C or C++
 * <> Indexing is column-based as in Matlab (not row-based as in C)
 * <> Use linear indexing.  [x*dimy+y] instead of [x][y]
 * Adapted from the code by Shawn Lankton (http://www.shawnlankton.com/2008/03/getting-started-with-mex-a-short-tutorial/)
 ********************************************************************/
#include <matrix.h>
#include <mex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm> 
using namespace std; 

/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
typedef int mwSize;
typedef int mwIndex;
typedef int mwSignedIndex;

#if (defined(_LP64) || defined(_WIN64)) && !defined(MX_COMPAT_32)
/* Currently 2^48 based on hardware limitations */
# define MWSIZE_MAX    281474976710655UL
# define MWINDEX_MAX   281474976710655UL
# define MWSINDEX_MAX  281474976710655L
# define MWSINDEX_MIN -281474976710655L
#else
# define MWSIZE_MAX    2147483647UL
# define MWINDEX_MAX   2147483647UL
# define MWSINDEX_MAX  2147483647L
# define MWSINDEX_MIN -2147483647L
#endif
#define MWSIZE_MIN    0UL
#define MWINDEX_MIN   0UL
#endif

void mexFunction(int nlmxhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
//declare variables
    mxArray *mxnl, *mxpart, *mxC, *mxCave, *mxS, *mxE, *mxIr, *mxIu, *mxpara, *mxbetamap, *mxalphamap;//input
    mxArray *mxeva_ori, *mxeva_dest, *mxNE, *mxSE, *mxEE, *mxIEr, *mxIEu, *mxevaids, *mxdestids;//input
    mxArray *mxnewS, *mxnewE, *mxnewIr, *mxnewIu, *mxdailyIr, *mxdailyIu;//output
    mxArray *mxnewSE, *mxnewEE, *mxnewIEr, *mxnewIEu, *mxdailyIEr, *mxdailyIEu;//output
    const mwSize *dims;
    double *nl, *part, *C, *Cave, *S, *E, *Ir, *Iu, *para, *betamap, *alphamap;//input
    double *eva_ori, *eva_dest, *NE, *SE, *EE, *IEr, *IEu, *evaids, *destids;//input
    double *newS, *newE, *newIr, *newIu, *dailyIr, *dailyIu;//output
    double *newSE, *newEE, *newIEr, *newIEu, *dailyIEr, *dailyIEu;//output
    int num_mp, num_loc, num_para, num_mpE, num_eva, num_dest;

//associate inputs
    mxnl = mxDuplicateArray(prhs[0]);
    mxpart = mxDuplicateArray(prhs[1]);
    mxC = mxDuplicateArray(prhs[2]);
    mxCave = mxDuplicateArray(prhs[3]);
    mxS = mxDuplicateArray(prhs[4]);
    mxE = mxDuplicateArray(prhs[5]);
    mxIr = mxDuplicateArray(prhs[6]);
    mxIu = mxDuplicateArray(prhs[7]);
    mxpara = mxDuplicateArray(prhs[8]);
    mxbetamap = mxDuplicateArray(prhs[9]);
    mxalphamap = mxDuplicateArray(prhs[10]);
    mxeva_ori = mxDuplicateArray(prhs[11]);
    mxeva_dest = mxDuplicateArray(prhs[12]);
    mxNE = mxDuplicateArray(prhs[13]);
    mxSE = mxDuplicateArray(prhs[14]);
    mxEE = mxDuplicateArray(prhs[15]);
    mxIEr = mxDuplicateArray(prhs[16]);
    mxIEu = mxDuplicateArray(prhs[17]);
    mxevaids = mxDuplicateArray(prhs[18]);
    mxdestids = mxDuplicateArray(prhs[19]);
    
//figure out dimensions
    dims = mxGetDimensions(prhs[0]);//number of subpopulation
    num_mp = (int)dims[0];
    dims = mxGetDimensions(prhs[1]);//number of locations
    num_loc = (int)dims[0]-1;
    dims = mxGetDimensions(prhs[8]);//number of parameters
    num_para = (int)dims[0];
    dims = mxGetDimensions(prhs[11]);//number of evacuated subpopulation
    num_mpE = (int)dims[0];
    dims = mxGetDimensions(prhs[18]);//number of evacuated locations
    num_eva = (int)dims[0];
    dims = mxGetDimensions(prhs[19]);//number of destination locations
    num_dest = (int)dims[0];
    
    
//associate outputs
    mxnewS = plhs[0] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    mxnewE = plhs[1] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    mxnewIr = plhs[2] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    mxnewIu = plhs[3] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    mxdailyIr = plhs[4] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    mxdailyIu = plhs[5] = mxCreateDoubleMatrix(num_mp,1,mxREAL);
    
    mxnewSE = plhs[6] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);
    mxnewEE = plhs[7] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);
    mxnewIEr = plhs[8] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);
    mxnewIEu = plhs[9] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);
    mxdailyIEr = plhs[10] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);
    mxdailyIEu = plhs[11] = mxCreateDoubleMatrix(num_mpE,1,mxREAL);

    
//associate pointers
    nl = mxGetPr(mxnl);
    part = mxGetPr(mxpart);
    C = mxGetPr(mxC);
    Cave = mxGetPr(mxCave);
    S = mxGetPr(mxS);
    E = mxGetPr(mxE);
    Ir = mxGetPr(mxIr);
    Iu = mxGetPr(mxIu);
    para = mxGetPr(mxpara);
    betamap = mxGetPr(mxbetamap);
    alphamap = mxGetPr(mxalphamap);
    
    eva_ori = mxGetPr(mxeva_ori);
    eva_dest = mxGetPr(mxeva_dest);
    NE = mxGetPr(mxNE);
    SE = mxGetPr(mxSE);
    EE = mxGetPr(mxEE);
    IEr = mxGetPr(mxIEr);
    IEu = mxGetPr(mxIEu);
    evaids = mxGetPr(mxevaids);
    destids = mxGetPr(mxdestids);
    
    newS = mxGetPr(mxnewS);
    newE = mxGetPr(mxnewE);
    newIr = mxGetPr(mxnewIr);
    newIu = mxGetPr(mxnewIu);
    dailyIr = mxGetPr(mxdailyIr);
    dailyIu = mxGetPr(mxdailyIu);
    
    newSE = mxGetPr(mxnewSE);
    newEE = mxGetPr(mxnewEE);
    newIEr = mxGetPr(mxnewIEr);
    newIEu = mxGetPr(mxnewIEu);
    dailyIEr = mxGetPr(mxdailyIEr);
    dailyIEu = mxGetPr(mxdailyIEu);
    
    ////////////////////////////////////////
    //do something
    default_random_engine generator((unsigned)time(NULL));
    //initialize auxillary variables
    int i, j;
    //para=Z,D,mu,theta,alpha1,alpha2,alpha3,beta0,beta1,...
    double Z=para[0],D=para[1],mu=para[2],theta=para[3];
    double dt1=(double)1/3, dt2=1-dt1;
    //daytime population,Ir,Iu,S,E,R
    vector<double> ND(num_loc),IrD(num_loc),IuD(num_loc),SD(num_loc),ED(num_loc),RD(num_loc);
    //daytime enter R, E, Iu
    vector<double> RentD(num_loc),EentD(num_loc),IuentD(num_loc);
    //nighttime population,Ir,Iu,S,E,R
    vector<double> NN(num_loc),IrN(num_loc),IuN(num_loc),SN(num_loc),EN(num_loc),RN(num_loc);
    //nighttime enter R, E, Iu
    vector<double> RentN(num_loc),EentN(num_loc),IuentN(num_loc);
    //total outgoing population
    vector<double> popleft(num_loc);
    //intermediate S, E, Iu, Ir, R
    vector<double> tempS(num_mp),tempE(num_mp),tempIu(num_mp),tempIr(num_mp),tempR(num_mp);
    //intermediate S, E, Iu, Ir, R for evaucated subpopulation
    vector<double> tempSE(num_mpE),tempEE(num_mpE),tempIEu(num_mpE),tempIEr(num_mpE),tempRE(num_mpE);
    //R and newR
    vector<double> R(num_mp), newR(num_mp);
    //RE and newRE
    vector<double> RE(num_mpE), newRE(num_mpE); 
    /////////////////////////////
    //change index in nl, part and ids (0-based index)
    for (i=0; i<num_mp; i++)
        nl[i]=nl[i]-1;
    for (i=0; i<num_loc+1; i++)
        part[i]=part[i]-1;
    for (i=0; i<num_loc; i++){
        betamap[i]=betamap[i]-1;
        alphamap[i]=alphamap[i]-1;
    }
    for (i=0; i<num_mpE; i++){
        eva_ori[i]=eva_ori[i]-1;
        eva_dest[i]=eva_dest[i]-1;
    }
    for (i=0; i<num_eva; i++)
        evaids[i]=evaids[i]-1;
    for (i=0; i<num_dest; i++)
        destids[i]=destids[i]-1;
    
    
    //is evacuated and destination
    vector<int> iseva(num_loc), isdest(num_loc);
    for (i=0; i<num_eva; i++)
        iseva[evaids[i]]=1;
    for (i=0; i<num_dest; i++)
        isdest[destids[i]]=1;
    //evacuated population VD,IErD,IEuD in destination
    vector<double> VD(num_loc),IErD(num_loc), IEuD(num_loc);
    for (i=0; i<num_mpE; i++){
        VD[(int)eva_dest[i]] += NE[i];
        IErD[(int)eva_dest[i]] += IEr[i];
        IEuD[(int)eva_dest[i]] += IEu[i];
    }
    
    
    //compute popleft
    for (i=0; i<num_loc; i++){
        for (j=part[i]+1; j<part[i+1]; j++){
            popleft[i]=popleft[i]+Cave[j];
        }
    }
    //assgn intermediate S, E, Iu, Ir, R
    for (i=0; i<num_mp; i++){
        R[i] = max(C[i]-S[i]-E[i]-Ir[i]-Iu[i],0.0);
        tempS[i] = S[i];
        tempE[i] = E[i];
        tempIr[i] = Ir[i];
        tempIu[i] = Iu[i];
        tempR[i] = R[i];
    }
    
    //assgn intermediate SE, EE, IEu, IEr, RE
    for (i=0; i<num_mpE; i++){
        RE[i] = max(NE[i]-SE[i]-EE[i]-IEr[i]-IEu[i],0.0);
        tempSE[i] = SE[i];
        tempEE[i] = EE[i];
        tempIEr[i] = IEr[i];
        tempIEu[i] = IEu[i];
        tempRE[i] = RE[i];
    }
    
    ///////////////////////////
    //daytime transmission
    //compute ND
    for (i=0; i<num_loc; i++){
        ND[i]=C[(int)part[i]];//i<-i
        for (j=part[i]+1; j<part[i+1]; j++){
            ND[i]=ND[i]+Ir[j];//reported infections (no mobility)
        }
    }
    for (i=0; i<num_loc; i++){
        if (iseva[i]==0){//not evacuated
            for (j=part[i]+1; j<part[i+1]; j++){
                ND[(int)nl[j]]=ND[(int)nl[j]]+C[j]-Ir[j];//commuting with reported infections removed
            }
        }
        if (iseva[i]!=1){//evacuated
            for (j=part[i]+1; j<part[i+1]; j++){
                ND[i]=ND[i]+C[j]-Ir[j];//commuting with reported infections removed
            }
        }
    }
    
    //comput IrD,IuD,SD,ED,RD
    for (i=0; i<num_loc; i++){
        if (iseva[i]==0){//not evacuated
            for (j=part[i]; j<part[i+1]; j++){
                IrD[i]=IrD[i]+Ir[j];
                IuD[(int)nl[j]]=IuD[(int)nl[j]]+Iu[j];
                SD[(int)nl[j]]=SD[(int)nl[j]]+S[j];
                ED[(int)nl[j]]=ED[(int)nl[j]]+E[j];
                RD[(int)nl[j]]=RD[(int)nl[j]]+R[j];
            }
        }
        if (iseva[i]==1){//evacuated
            for (j=part[i]; j<part[i+1]; j++){
                IrD[i]=IrD[i]+Ir[j];
                IuD[i]=IuD[i]+Iu[j];
                SD[i]=SD[i]+S[j];
                ED[i]=ED[i]+E[j];
                RD[i]=RD[i]+R[j];
            }
        }
    }
    
    //compute RentD, EentD and IuentD
    for (i=0; i<num_loc; i++){
        for (j=part[i]+1; j<part[i+1]; j++){
            RentD[(int)nl[j]]=RentD[(int)nl[j]]+Cave[j]*RD[i]/(ND[i]-IrD[i]);
            EentD[(int)nl[j]]=EentD[(int)nl[j]]+Cave[j]*ED[i]/(ND[i]-IrD[i]);
            IuentD[(int)nl[j]]=IuentD[(int)nl[j]]+Cave[j]*IuD[i]/(ND[i]-IrD[i]);
        }
    }
    //compute for each subpopulation
    for (i=0; i<num_loc; i++){
        for (j=part[i]; j<part[i+1]; j++){
            double Eexpr;
            double Eexpu;
            //////////////////////////////
            if (iseva[i]==0){//not evacuated
                double beta=para[(int)betamap[(int)nl[j]]];
                Eexpr=beta*S[j]*(IrD[(int)nl[j]]+IErD[(int)nl[j]])/(ND[(int)nl[j]]+VD[(int)nl[j]])*dt1;//new exposed due to reported cases
                Eexpu=mu*beta*S[j]*(IuD[(int)nl[j]]+IEuD[(int)nl[j]])/(ND[(int)nl[j]]+VD[(int)nl[j]])*dt1;//new exposed due to unreported cases
            }
            if (iseva[i]!=0){
                double beta=para[(int)betamap[i]];
                Eexpr=beta*S[j]*(IrD[i]+IErD[i])/(ND[i]+VD[i])*dt1;//new exposed due to reported cases
                Eexpu=mu*beta*S[j]*(IuD[i]+IEuD[i])/(ND[i]+VD[i])*dt1;//new exposed due to unreported cases
            }
            double alpha=para[(int)alphamap[i]];
            double Einfr=alpha*E[j]/Z*dt1;//new reported cases
            double Einfu=(1-alpha)*E[j]/Z*dt1;//new unreported cases
            double Erecr=Ir[j]/D*dt1;//new recovery of reported cases
            double Erecu=Iu[j]/D*dt1;//new recovery of unreported cases
            ///////////////////////////
            double ERenter=theta*dt1*(C[j]-Ir[j])/ND[(int)nl[j]]*RentD[(int)nl[j]];//incoming R
            double ERleft=theta*dt1*R[j]/(ND[(int)nl[j]]-IrD[(int)nl[j]])*popleft[(int)nl[j]];//outgoing R
            double EEenter=theta*dt1*(C[j]-Ir[j])/ND[(int)nl[j]]*EentD[(int)nl[j]];//incoming E
            double EEleft=theta*dt1*E[j]/(ND[(int)nl[j]]-IrD[(int)nl[j]])*popleft[(int)nl[j]];//outgoing E
            double EIuenter=theta*dt1*(C[j]-Ir[j])/ND[(int)nl[j]]*IuentD[(int)nl[j]];//incoming Iu
            double EIuleft=theta*dt1*Iu[j]/(ND[(int)nl[j]]-IrD[(int)nl[j]])*popleft[(int)nl[j]];//outgoing Iu
            
            ////////////////////
            //stochastic: poisson
            poisson_distribution<int> distribution1(Eexpr);
            int expr=min(distribution1(generator),(int)(S[j]*dt1));
            poisson_distribution<int> distribution2(Eexpu);
            int expu=min(distribution2(generator),(int)(S[j]*dt1));
            poisson_distribution<int> distribution3(Einfr);
            int infr=min(distribution3(generator),(int)(E[j]*dt1));
            poisson_distribution<int> distribution4(Einfu);
            int infu=min(distribution4(generator),(int)(E[j]*dt1));
            poisson_distribution<int> distribution5(Erecr);
            int recr=min(distribution5(generator),(int)(Ir[j]*dt1));
            poisson_distribution<int> distribution6(Erecu);
            int recu=min(distribution6(generator),(int)(Iu[j]*dt1));
            poisson_distribution<int> distribution7(ERenter);
            int Renter=distribution7(generator);
            poisson_distribution<int> distribution8(ERleft);
            int Rleft=min(distribution8(generator),(int)(R[j]*dt1));
            poisson_distribution<int> distribution9(EEenter);
            int Eenter=distribution9(generator);
            poisson_distribution<int> distribution10(EEleft);
            int Eleft=min(distribution10(generator),(int)(E[j]*dt1));
            poisson_distribution<int> distribution11(EIuenter);
            int Iuenter=distribution11(generator);
            poisson_distribution<int> distribution12(EIuleft);
            int Iuleft=min(distribution12(generator),(int)(Iu[j]*dt1));
            
            /////////////////////
            tempR[j]=max((int)(tempR[j]+recr+recu+Renter-Rleft),0);
            tempE[j]=max((int)(tempE[j]+expr+expu-infr-infu+Eenter-Eleft),0);
            tempIr[j]=max((int)(tempIr[j]+infr-recr),0);
            tempIu[j]=max((int)(tempIu[j]+infu-recu+Iuenter-Iuleft),0);
            dailyIr[j]=max((int)(dailyIr[j]+infr),0);
            dailyIu[j]=max((int)(dailyIu[j]+infu),0);
            tempS[j]=max((int)(C[j]-tempE[j]-tempIr[j]-tempIu[j]-tempR[j]),0);
        }
    }
    
    //compute for each evacuated subpopulation
    for (i=0; i<num_mpE; i++){
        int dest = (int)eva_dest[i];
        int ori = (int)eva_ori[i];
        //////////////////////////////
        double beta=para[(int)betamap[dest]];
        double alpha=para[(int)alphamap[ori]];
        double Eexpr=beta*SE[i]*(IrD[dest]+IErD[dest])/(ND[dest]+VD[dest])*dt1;//new exposed due to reported cases
        double Eexpu=mu*beta*SE[i]*(IuD[dest]+IEuD[dest])/(ND[dest]+VD[dest])*dt1;//new exposed due to unreported cases

        double Einfr=alpha*EE[i]/Z*dt1;//new reported cases
        double Einfu=(1-alpha)*EE[i]/Z*dt1;//new unreported cases
        double Erecr=IEr[i]/D*dt1;//new recovery of reported cases
        double Erecu=IEu[i]/D*dt1;//new recovery of unreported cases

        ////////////////////
        //stochastic: poisson
        poisson_distribution<int> distribution1(Eexpr);
        int expr=min(distribution1(generator),(int)(SE[i]*dt1));
        poisson_distribution<int> distribution2(Eexpu);
        int expu=min(distribution2(generator),(int)(SE[i]*dt1));
        poisson_distribution<int> distribution3(Einfr);
        int infr=min(distribution3(generator),(int)(EE[i]*dt1));
        poisson_distribution<int> distribution4(Einfu);
        int infu=min(distribution4(generator),(int)(EE[i]*dt1));
        poisson_distribution<int> distribution5(Erecr);
        int recr=min(distribution5(generator),(int)(IEr[i]*dt1));
        poisson_distribution<int> distribution6(Erecu);
        int recu=min(distribution6(generator),(int)(IEu[i]*dt1));

        /////////////////////
        tempRE[i]=max((int)(tempRE[i]+recr+recu),0);
        tempEE[i]=max((int)(tempEE[i]+expr+expu-infr-infu),0);
        tempIEr[i]=max((int)(tempIEr[i]+infr-recr),0);
        tempIEu[i]=max((int)(tempIEu[i]+infu-recu),0);
        dailyIEr[i]=max((int)(dailyIEr[i]+infr),0);
        dailyIEu[i]=max((int)(dailyIEu[i]+infu),0);
        tempSE[i]=max((int)(NE[i]-tempEE[i]-tempIEr[i]-tempIEu[i]-tempRE[i]),0);

    }
    
    ////////////////////////////////
    //nighttime transmission
    //assgn new S, E, Iu, Ir, R
    for (i=0; i<num_mp; i++){
        newS[i] = tempS[i];
        newE[i] = tempE[i];
        newIr[i] = tempIr[i];
        newIu[i] = tempIu[i];
        newR[i] = tempR[i];
    }
    
    //assgn new SE, EE, IEu, IEr, RE
    for (i=0; i<num_mpE; i++){
        newSE[i] = tempSE[i];
        newEE[i] = tempEE[i];
        newIEr[i] = tempIEr[i];
        newIEu[i] = tempIEu[i];
        newRE[i] = tempRE[i];
    }
    
    //compute NN
    for (i=0; i<num_loc; i++){
        for (j=part[i]; j<part[i+1]; j++){
            NN[i]=NN[i]+C[j];
        }
    }
    
    //comput IrN,IuN,SN,EN,RN
    for (i=0; i<num_loc; i++){
        for (j=part[i]; j<part[i+1]; j++){
            IrN[i]=IrN[i]+tempIr[j];
            IuN[i]=IuN[i]+tempIu[j];
            SN[i]=SN[i]+tempS[j];
            EN[i]=EN[i]+tempE[j];
            RN[i]=RN[i]+tempR[j];
        }
    }
    //compute RentN, EentN and IuentN
    for (i=0; i<num_loc; i++){
        for (j=part[i]+1; j<part[i+1]; j++){
            RentN[(int)nl[j]]=RentN[(int)nl[j]]+Cave[j]*RN[i]/(NN[i]-IrN[i]);
            EentN[(int)nl[j]]=EentN[(int)nl[j]]+Cave[j]*EN[i]/(NN[i]-IrN[i]);
            IuentN[(int)nl[j]]=IuentN[(int)nl[j]]+Cave[j]*IuN[i]/(NN[i]-IrN[i]);
        }
    }
    //compute for each subpopulation
    for (i=0; i<num_loc; i++){
        for (j=part[i]; j<part[i+1]; j++){
            //////////////////////////////
            double beta=para[(int)betamap[i]];
            double alpha=para[(int)alphamap[i]];
            double Eexpr=beta*tempS[j]*(IrN[i]+IErD[i])/(NN[i]+VD[i])*dt2;//new exposed due to reported cases
            double Eexpu=mu*beta*tempS[j]*(IuN[i]+IEuD[i])/(NN[i]+VD[i])*dt2;//new exposed due to unreported cases
            double Einfr=alpha*tempE[j]/Z*dt2;//new reported cases
            double Einfu=(1-alpha)*tempE[j]/Z*dt2;//new unreported cases
            double Erecr=tempIr[j]/D*dt2;//new recovery of reported cases
            double Erecu=tempIu[j]/D*dt2;//new recovery of unreported cases
            ///////////////////////////
            double ERenter=theta*dt2*C[j]/NN[i]*RentN[i];//incoming R
            double ERleft=theta*dt2*tempR[j]/(NN[i]-IrN[i])*popleft[i];//outgoing R
            double EEenter=theta*dt2*C[j]/NN[i]*EentN[i];//incoming E
            double EEleft=theta*dt2*tempE[j]/(NN[i]-IrN[i])*popleft[i];//outgoing E
            double EIuenter=theta*dt2*C[j]/NN[i]*IuentN[i];//incoming Iu
            double EIuleft=theta*dt2*tempIu[j]/(NN[i]-IrN[i])*popleft[i];//outgoing Iu
            ////////////////////
            //stochastic: poisson
            poisson_distribution<int> distribution1(Eexpr);
            int expr=min(distribution1(generator),(int)(tempS[j]*dt2));
            poisson_distribution<int> distribution2(Eexpu);
            int expu=min(distribution2(generator),(int)(tempS[j]*dt2));
            poisson_distribution<int> distribution3(Einfr);
            int infr=min(distribution3(generator),(int)(tempE[j]*dt2));
            poisson_distribution<int> distribution4(Einfu);
            int infu=min(distribution4(generator),(int)(tempE[j]*dt2));
            poisson_distribution<int> distribution5(Erecr);
            int recr=min(distribution5(generator),(int)(tempIr[j]*dt2));
            poisson_distribution<int> distribution6(Erecu);
            int recu=min(distribution6(generator),(int)(tempIu[j]*dt2));
            poisson_distribution<int> distribution7(ERenter);
            int Renter=distribution7(generator);
            poisson_distribution<int> distribution8(ERleft);
            int Rleft=min(distribution8(generator),(int)(tempR[j]*dt2));
            poisson_distribution<int> distribution9(EEenter);
            int Eenter=distribution9(generator);
            poisson_distribution<int> distribution10(EEleft);
            int Eleft=min(distribution10(generator),(int)(tempE[j]*dt2));
            poisson_distribution<int> distribution11(EIuenter);
            int Iuenter=distribution11(generator);
            poisson_distribution<int> distribution12(EIuleft);
            int Iuleft=min(distribution12(generator),(int)(tempIu[j]*dt2));
            /////////////////////
            newR[j]=max((int)(newR[j]+recr+recu+Renter-Rleft),0);
            newE[j]=max((int)(newE[j]+expr+expu-infr-infu+Eenter-Eleft),0);
            newIr[j]=max((int)(newIr[j]+infr-recr),0);
            newIu[j]=max((int)(newIu[j]+infu-recu+Iuenter-Iuleft),0);
            dailyIr[j]=max((int)(dailyIr[j]+infr),0);
            dailyIu[j]=max((int)(dailyIu[j]+infu),0);
            newS[j]=max((int)(C[j]-newE[j]-newIr[j]-newIu[j]-newR[j]),0);
        }
    }
    
    //compute for each evacuated subpopulation
    for (i=0; i<num_mpE; i++){
        int dest = (int)eva_dest[i];
        int ori = (int)eva_ori[i];
        //////////////////////////////
        double beta=para[(int)betamap[dest]];
        double alpha=para[(int)alphamap[ori]];
        double Eexpr=beta*tempSE[i]*(IrN[dest]+IErD[dest])/(NN[dest]+VD[dest])*dt2;//new exposed due to reported cases
        double Eexpu=mu*beta*tempSE[i]*(IuN[dest]+IEuD[dest])/(NN[dest]+VD[dest])*dt2;//new exposed due to unreported cases
        double Einfr=alpha*tempEE[i]/Z*dt2;//new reported cases
        double Einfu=(1-alpha)*tempEE[i]/Z*dt2;//new unreported cases
        double Erecr=tempIEr[i]/D*dt2;//new recovery of reported cases
        double Erecu=tempIEu[i]/D*dt2;//new recovery of unreported cases

        ////////////////////
        //stochastic: poisson
        poisson_distribution<int> distribution1(Eexpr);
        int expr=min(distribution1(generator),(int)(tempSE[i]*dt2));
        poisson_distribution<int> distribution2(Eexpu);
        int expu=min(distribution2(generator),(int)(tempSE[i]*dt2));
        poisson_distribution<int> distribution3(Einfr);
        int infr=min(distribution3(generator),(int)(tempEE[i]*dt2));
        poisson_distribution<int> distribution4(Einfu);
        int infu=min(distribution4(generator),(int)(tempEE[i]*dt2));
        poisson_distribution<int> distribution5(Erecr);
        int recr=min(distribution5(generator),(int)(tempIEr[i]*dt2));
        poisson_distribution<int> distribution6(Erecu);
        int recu=min(distribution6(generator),(int)(tempIEu[i]*dt2));
        /////////////////////
        newRE[i]=max((int)(newRE[i]+recr+recu),0);
        newEE[i]=max((int)(newEE[i]+expr+expu-infr-infu),0);
        newIEr[i]=max((int)(newIEr[i]+infr-recr),0);
        newIEu[i]=max((int)(newIEu[i]+infu-recu),0);
        dailyIEr[i]=max((int)(dailyIEr[i]+infr),0);
        dailyIEu[i]=max((int)(dailyIEu[i]+infu),0);
        newSE[i]=max((int)(NE[i]-newEE[i]-newIEr[i]-newIEu[i]-newRE[i]),0);
    }
    
    /////////////////////
    return;
}
