function EvacuationSimulator()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code and data for the following manuscript
%Pei S., Dahl K., Yamana T., Licker R., Shaman J. 
%Compound risks of hurricane evacuation amid the COVID-19 pandemic in the United States
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%compile the cpp function in matlab before use
%mex model_eakf.cpp
%mex model_eva.cpp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load dailyincidence%county-level incidence data
load dailydeaths%county-level death data
num_times=size(dailyincidence,2);%total length of data
num_ens=100;%number of ensemble members
%evacuation setting
load evacuation_normal%baseline scenario
%evaids: origin; destids: destination; V: evacuation matrix
%Vij is the number of evacuees from origin j to destination i
Tstart=datetime('21/02/20','InputFormat','dd/MM/yy');%start of disease data
Tprojection=Tstart+num_times-1;%start of projection
preeva=3;%pre-evacuation
posteva=3;%post-evacuation
L_eva=7;%evacuation for 7 days
origin_transmissionrate=0.2;%origin transmission rate increased by 20% (pre, during, post evacuation)
dest_transmissionrate=0.1;%destination transmission rate increased by 10% (during evacuation)
aftereva=2*7;%simulate 14 days after posteva
T=num_times+preeva+L_eva+posteva+aftereva;%total simulation duration
Teva=num_times+preeva;%start of evacuation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Td=9;%average reporting delay
a=1.85;%shape parameter of gamma distribution
b=Td/a;%scale parameter of gamma distribution
rnds=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers
Td_death=9+7;%average delay of death
a=1.85;%shape parameter of gamma distribution
b=Td_death/a;%scale parameter of gamma distribution
rnds_d=ceil(gamrnd(a,b,1e4,1));%pre-generate gamma random numbers
load commutedata
load population
%%%%%%%%%%%%%%%
%Inter-county commuting is stored using neighbor list: nl, part and C. 
%nl (neighbor list) and part (partition) both describe the network structure. 
%For instance, the neighbors of location i are nl(part(i):part(i+1)-1). 
%This neighbor set include i itself. 
%C has the same size with nl. 
%The commuters from location i to location j=nl(part(i)+x) is C(part(i)+x).
%%%%%%%%%%%%%%%
num_loc=size(part,1)-1;%number of counties
num_mp=size(nl,1);%number of subpopulations
obs_case=zeros(size(dailyincidence));
obs_death=zeros(size(dailydeaths));
%smooth the data: 7 day moving average
for l=1:num_loc
    for t=1:num_times
        if (t+3)<=num_times
            obs_case(l,t)=mean(dailyincidence(l,max(1,t-3):min(t+3,num_times)));
            obs_death(l,t)=mean(dailydeaths(l,max(1,t-3):min(t+3,num_times)));
        else
            obs_case(l,t)=mean(dailyincidence(l,max(1,num_times-6):num_times));
            obs_death(l,t)=mean(dailydeaths(l,max(1,num_times-6):num_times));
        end
    end
end
load deathrate_IFR %IFR for each county, averaged based on age structure

incidence=dailyincidence;%incidence for seeding
%set OEV
OEV_case=zeros(size(dailyincidence));
OEV_death=zeros(size(dailydeaths));
for l=1:num_loc
    for t=1:num_times
        obs_ave=mean(dailyincidence(l,max(1,t-6):t));
        OEV_case(l,t)=max(1,obs_ave^2/25);
        death_ave=mean(dailydeaths(l,max(1,t-6):t));
        OEV_death(l,t)=max(1,death_ave^2/100);
    end
end
%adjusting inter-county movement
load MI_inter
%adjusting mobility starting from March 16, day 25
MI_inter=MI_inter(:,2:end);
MI_inter_relative=MI_inter;
for t=25:size(MI_inter_relative,2)
    MI_inter_relative(:,t)=MI_inter(:,t)./MI_inter(:,t-1);
    MI_inter_relative(isnan(MI_inter_relative(:,t)),t)=0;
    MI_inter_relative(isinf(MI_inter_relative(:,t)),t)=0;
    MI_inter_relative(:,t)=min(MI_inter_relative(:,t),1);
end
MI_inter_relative(:,1:24)=1;

C=C*ones(1,T);
Cave=Cave*ones(1,T);
%adjusting mobility starting from March 16, day 25
for t=25:num_times%day 25: March 16
    C(:,t)=C(:,t-1);
    Cave(:,t)=Cave(:,t-1);
    for l=1:num_loc
        for j=part(l)+1:part(l+1)-1
            if t<=size(MI_inter_relative,2)
                C(part(l),t)=C(part(l),t)+((1-MI_inter_relative(nl(j),t))*C(j,t));
                Cave(part(l),t)=Cave(part(l),t)+((1-MI_inter_relative(nl(j),t))*Cave(j,t));
                C(j,t)=(MI_inter_relative(nl(j),t)*C(j,t));
                Cave(j,t)=(MI_inter_relative(nl(j),t)*Cave(j,t));
            end
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[S,E,Ir,Iu,Seedc]=initialize(nl,part,C(:,1),num_ens,incidence);
%%%%%%%%%%%%%%%%%%%%
%S,E,Ir and Iu represent susceptible, exposed, reported infection,
%unreported infection in all subpopulations.
%%%%%%%%%%%%%%%%%%%%
obs_temp=zeros(num_loc,num_ens,T);%records of reported cases
death_temp=zeros(num_loc,num_ens,T);%records of death
load('parafit1')% prior parameter setting, estimated using data until March 13 in a previous work
%initialize parameters
[para,paramax,paramin,betamap,alphamaps]=initializepara_eakf(dailyincidence,num_ens,parafit);
paramax_ori=paramax;
paramin_ori=paramin;
para_ori=para;
%%%%%%%%%%%%inflate parameters
lambda=2;
para=mean(para,2)*ones(1,num_ens)+lambda*(para-mean(para,2)*ones(1,num_ens));
para=checkbound_para(para,paramax,paramin);
%fix Z, D and theta
%Z
para(1,:)=parafit(3,1:num_ens);
%D
para(2,:)=parafit(4,1:num_ens);
%mu
para(3,:)=parafit(2,1:num_ens);
%theta
para(4,:)=parafit(6,1:num_ens);
parastd=std(para,0,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%calibrate the model using EAKF
lambda=1.2;%inflation to avoid ensemble collapse
num_para=size(para,1);
para_post=zeros(num_para,num_ens,T);%posterior parameters
S_post=zeros(num_loc,num_ens,T);%posterior susceptiblilty
for t=1:num_times%start from Feb 21
    t
    tic
    %fix Z, D and theta
    %Z
    para(1,:)=parafit(3,1:num_ens);
    %D
    para(2,:)=parafit(4,1:num_ens);
    %mu
    para(3,:)=parafit(2,1:num_ens);
    %theta
    para(4,:)=parafit(6,1:num_ens);
    %seeding
    if t<=size(Seedc,2)
        [S,E,Ir,Iu]=seeding(S,E,Ir,Iu,nl,part,C(:,t),Seedc,t);
    end
    %re-initialize if there is a second peak
    for l=1:num_loc
        if Seedc(l,t)>0
            para(betamap(l),:)=para_ori(betamap(l),:);
        end
    end
    %%%%%%%%%%%%%%%%%%
    %record model variables
    S_temp=S; E_temp=E; Ir_temp=Ir; Iu_temp=Iu;
    %integrate forward one step
    dailyIr_prior=zeros(num_mp,num_ens);
    dailyIu_prior=zeros(num_mp,num_ens);
    for k=1:num_ens%run for each ensemble member
        %adjust population according to change of inter-county movement
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k)]=adjustmobility(S(:,k),E(:,k),Ir(:,k),Iu(:,k),nl,part,MI_inter_relative,t);
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,dailyIu_temp]=model_eakf(nl,part,C(:,t),Cave(:,t),S(:,k),E(:,k),Ir(:,k),Iu(:,k),para(:,k),betamap,alphamaps);
        dailyIr_prior(:,k)=dailyIr_temp;
        dailyIu_prior(:,k)=dailyIu_temp;
    end
    %%%%%%%%%%%%%%%%%%%%%%
    %mini-forecast based on current model state
    %integrate forward for 6 days, prepare for observation
    Tproj=min(6,num_times-t);
    obs_temp1=obs_temp;
    death_temp1=death_temp;
    for t1=t:t+Tproj
        for k=1:num_ens%run for each ensemble member
            [S_temp(:,k),E_temp(:,k),Ir_temp(:,k),Iu_temp(:,k)]=adjustmobility(S_temp(:,k),E_temp(:,k),Ir_temp(:,k),Iu_temp(:,k),nl,part,MI_inter_relative,t1);
            [S_temp(:,k),E_temp(:,k),Ir_temp(:,k),Iu_temp(:,k),dailyIr_temp,dailyIu_temp]=model_eakf(nl,part,C(:,t1),Cave(:,t1),S_temp(:,k),E_temp(:,k),Ir_temp(:,k),Iu_temp(:,k),para(:,k),betamap,alphamaps);
            %reporting delay
            for l=1:num_loc
                for j=part(l):part(l+1)-1
                    inci=round(dailyIr_temp(j));
                    if inci>0
                        rnd=datasample(rnds,inci);
                        for h=1:length(rnd)
                            if (t1+rnd(h)<=T)
                                obs_temp1(l,k,t1+rnd(h))=obs_temp1(l,k,t1+rnd(h))+1;
                            end
                        end 
                    end
                end
            end
            %death delay
            for l=1:num_loc
                for j=part(l):part(l+1)-1
                    inci=round((dailyIr_temp(j)+dailyIu_temp(j))*deathrate(l,min(t1,size(deathrate,2))));
                    if inci>0
                        rnd=datasample(rnds_d,inci);
                        for h=1:length(rnd)
                            if (t1+rnd(h)<=T)
                                death_temp1(l,k,t1+rnd(h))=death_temp1(l,k,t1+rnd(h))+1;
                            end
                        end 
                    end
                end
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update model using death 6 days ahead
    t1=min(t+Tproj,num_times);
    death_ens=death_temp1(:,:,t1);%death at t1, prior
    %loop through local observations
    for l=1:num_loc
        %%%%%%%%%%%%%%%%%%%death
        %Get the variance of the ensemble
        obs_var = OEV_death(l,t1);
        prior_var = var(death_ens(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0%if degenerate
            post_var=1e-3;
            prior_var=1e-3;
        end
        prior_mean = mean(death_ens(l,:));
        post_mean = post_var*(prior_mean/prior_var + obs_death(l,t1)/obs_var);
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(death_ens(l,:)-prior_mean)-death_ens(l,:);
        %Loop over each state variable (connected to location l)
        %adjust related metapopulation
        neighbors=part(l):part(l+1)-1;%metapopulation live in l
        neighbors1=find(nl==l);%%metapopulation work in l
        neighbors=union(neighbors1,neighbors);
        for h=1:length(neighbors)
            j=neighbors(h);
            %E
            temp=E(j,:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            E(j,:)=E(j,:)+dx;
            %Ir
            temp=Ir(j,:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            Ir(j,:)=Ir(j,:)+dx;
            %Iu
            temp=Iu(j,:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            Iu(j,:)=Iu(j,:)+dx;
            %dailyIr
            temp=dailyIr_prior(j,:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            dailyIr_prior(j,:)=(max(dailyIr_prior(j,:)+dx,0));
            %dailyIu
            temp=dailyIu_prior(j,:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            dailyIu_prior(j,:)=(max(dailyIu_prior(j,:)+dx,0));
        end
        %adjust alpha before running out of observations of case and death
        if (t+6<=num_times)
            temp=para(alphamaps(l),:);
            A=cov(temp,death_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            para(alphamaps(l),:)=para(alphamaps(l),:)+dx;
            %inflation
            if std(para(alphamaps(l),:))<parastd(alphamaps(l))
                para(alphamaps(l),:)=mean(para(alphamaps(l),:),2)*ones(1,num_ens)+lambda*(para(alphamaps(l),:)-mean(para(alphamaps(l),:),2)*ones(1,num_ens));
            end
        end
        %adjust beta
        temp=para(betamap(l),:);
        A=cov(temp,death_ens(l,:));
        rr=A(2,1)/prior_var;
        dx=rr*dy;
        para(betamap(l),:)=para(betamap(l),:)+dx;
        %inflation
        if std(para(betamap(l),:))<parastd(betamap(l))
            para(betamap(l),:)=mean(para(betamap(l),:),2)*ones(1,num_ens)+lambda*(para(betamap(l),:)-mean(para(betamap(l),:),2)*ones(1,num_ens));
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update model using case 6 days ahead
    t1=min(t+6,num_times);
    obs_ens=obs_temp1(:,:,t1);%observation at t1, prior
    for l=1:num_loc
        %%%%%%%%%%%%%%%%%%%case
        %Get the variance of the ensemble
        obs_var = OEV_case(l,t1);
        prior_var = var(obs_ens(l,:));
        post_var = prior_var*obs_var/(prior_var+obs_var);
        if prior_var==0%if degenerate
            post_var=1e-3;
            prior_var=1e-3;
        end
        prior_mean = mean(obs_ens(l,:));
        post_mean = post_var*(prior_mean/prior_var + obs_case(l,t1)/obs_var);
        %%%% Compute alpha and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);
        %Loop over each state variable (connected to location l)
        %adjust related metapopulation
        neighbors=part(l):part(l+1)-1;%metapopulation live in l
        neighbors1=find(nl==l);%%metapopulation work in l
        neighbors=union(neighbors1,neighbors);
        for h=1:length(neighbors)
            j=neighbors(h);
            %E
            temp=E(j,:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            E(j,:)=E(j,:)+dx;
            %Ir
            temp=Ir(j,:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            Ir(j,:)=Ir(j,:)+dx;
            %Iu
            temp=Iu(j,:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            Iu(j,:)=Iu(j,:)+dx;
            %dailyIr
            temp=dailyIr_prior(j,:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            dailyIr_prior(j,:)=(max(dailyIr_prior(j,:)+dx,0));
            %dailyIu
            temp=dailyIu_prior(j,:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            dailyIu_prior(j,:)=(max(dailyIu_prior(j,:)+dx,0));
        end
        %adjust alpha
        if (t+6<=num_times)
            temp=para(alphamaps(l),:);
            A=cov(temp,obs_ens(l,:));
            rr=A(2,1)/prior_var;
            dx=rr*dy;
            para(alphamaps(l),:)=para(alphamaps(l),:)+dx;
            %inflation
            if std(para(alphamaps(l),:))<parastd(alphamaps(l))
                para(alphamaps(l),:)=mean(para(alphamaps(l),:),2)*ones(1,num_ens)+lambda*(para(alphamaps(l),:)-mean(para(alphamaps(l),:),2)*ones(1,num_ens));
            end
        end
        %adjust beta
        temp=para(betamap(l),:);
        A=cov(temp,obs_ens(l,:));
        rr=A(2,1)/prior_var;
        dx=rr*dy;
        para(betamap(l),:)=para(betamap(l),:)+dx;
        %inflation if ensemble spread is narrow
        if std(para(betamap(l),:))<parastd(betamap(l))
            para(betamap(l),:)=mean(para(betamap(l),:),2)*ones(1,num_ens)+lambda*(para(betamap(l),:)-mean(para(betamap(l),:),2)*ones(1,num_ens));
        end
    end
    para=checkbound_para(para,paramax,paramin);
    %update Ir and Iu posterior
    dailyIr_post=dailyIr_prior;
    dailyIu_post=dailyIu_prior;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %update observations
    for k=1:num_ens
        %update obs_temp
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=round(dailyIr_post(j,k));
                if inci>0
                    rnd=datasample(rnds,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            obs_temp(l,k,t+rnd(h))=obs_temp(l,k,t+rnd(h))+1;
                        end
                    end 
                end
            end
        end
        %update death_temp
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=round((dailyIr_post(j,k)+dailyIu_post(j,k))*deathrate(l,min(t,size(deathrate,2))));
                if inci>0
                    rnd=datasample(rnds_d,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            death_temp(l,k,t+rnd(h))=death_temp(l,k,t+rnd(h))+1;
                        end
                    end 
                end
            end
        end
    end
    [S,E,Ir,Iu]=checkbound(S,E,Ir,Iu);
    para_post(:,:,t)=para;
    for l=1:num_loc
        S_post(l,:,t)=min(1,sum(S(part(l):part(l+1)-1,:))/population(l));
    end
   toc
end

%%%%%%%%%%%store intermideiate variables
S_start=S; E_start=E; Ir_start=Ir; Iu_start=Iu;
obs_temp_start=obs_temp;%records of reported cases

disp('Start simulations with evacuation')
%Simulation with evacuation
S=S_start; E=E_start; Iu=Iu_start; Ir=Ir_start;
obs_proj=obs_temp_start;
parapost=para_post(:,:,num_times);
paratemp=parapost;

%increase beta in origin
paratemp(betamap(evaids),:)=paratemp(betamap(evaids),:)*(1+origin_transmissionrate);

for t=num_times+1:num_times+preeva
    t
    tic
    for k=1:num_ens%run for each ensemble member
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,~]=model_eakf(nl,part,C(:,min(t,size(C,2))),Cave(:,min(t,size(Cave,2))),S(:,k),E(:,k),Ir(:,k),Iu(:,k),paratemp(:,k),betamap,alphamaps);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
                if inci>0
                    rnd=datasample(rnds,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            obs_proj(l,k,t+rnd(h))=obs_proj(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
    end
    toc
end

%%%%%%%%%%%%%%%%%%%%
%increase beta in destination

acceptedevacuees=sum(V,2);
id=destids(acceptedevacuees>0);
id=setdiff(id,evaids);

paratemp(betamap(id),:)=paratemp(betamap(id),:)*(1+dest_transmissionrate);
%%%%%%%%%%%%%%%%%%%%

%set evacuation subpopulation
[C1,Cave1,eva_ori,eva_dest,NE,SE,EE,IEr,IEu,S,E,Ir,Iu]=getC_eva(C(:,min(Teva,size(C,2))),Cave(:,min(Teva,size(Cave,2))),nl,part,S,E,Ir,Iu,V,evaids,destids);
num_mpE=size(NE,1);

for t=num_times+preeva+1:num_times+preeva+L_eva
    t
    tic
    for k=1:num_ens
        %no need to adjust cross-county mobility
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,~,SE(:,k),EE(:,k),IEr(:,k),IEu(:,k),dailyIEr_temp,dailyIEu_temp]=...
            model_eva(nl,part,C1,Cave1,S(:,k),E(:,k),Ir(:,k),Iu(:,k),paratemp(:,k),betamap,alphamaps,eva_ori,eva_dest,NE,SE(:,k),EE(:,k),IEr(:,k),IEu(:,k),evaids,destids);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
                if inci>0
                    rnd=datasample(rnds,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            obs_proj(l,k,t+rnd(h))=obs_proj(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
        %add evacuees to destination counties
        %reporting delay
        for i=1:num_mpE
            inci=dailyIEr_temp(i);
            if inci>0
                rnd=datasample(rnds,inci);
                for h=1:length(rnd)
                    if (t+rnd(h)<=T)
                        if (t+rnd(h)<=num_times+preeva+L_eva)
                            l=eva_dest(i);
                        else
                            l=eva_ori(i);
                        end
                        obs_proj(l,k,t+rnd(h))=obs_proj(l,k,t+rnd(h))+1;
                    end
                end
            end
        end
    end
    toc
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%decrease beta in destination
paratemp(betamap(id),:)=paratemp(betamap(id),:)*(1-dest_transmissionrate);
%%%%%%%%%%%%%%%%%%%%%%%%%%

%return
[S,E,Ir,Iu]=returnhome(C(:,min(Teva+L_eva,size(C,2))),part,S,E,Ir,Iu,eva_ori,SE,EE,IEr,IEu);

for t=num_times+preeva+L_eva+1:num_times+preeva+L_eva+posteva
    t
    tic
    for k=1:num_ens
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,~]=model_eakf(nl,part,C(:,min(t,size(C,2))),Cave(:,min(t,size(Cave,2))),S(:,k),E(:,k),Ir(:,k),Iu(:,k),paratemp(:,k),betamap,alphamaps);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
                if inci>0
                    rnd=datasample(rnds,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            obs_proj(l,k,t+rnd(h))=obs_proj(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
    end
    toc
end

%decrease beta in origin
paratemp(betamap(evaids),:)=paratemp(betamap(evaids),:)*(1-origin_transmissionrate);

for t=num_times+preeva+L_eva+posteva+1:T
    t
    tic
    for k=1:num_ens
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,~]=model_eakf(nl,part,C(:,min(t,size(C,2))),Cave(:,min(t,size(Cave,2))),S(:,k),E(:,k),Ir(:,k),Iu(:,k),paratemp(:,k),betamap,alphamaps);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
                if inci>0
                    rnd=datasample(rnds,inci);
                    for h=1:length(rnd)
                        if (t+rnd(h)<=T)
                            obs_proj(l,k,t+rnd(h))=obs_proj(l,k,t+rnd(h))+1;
                        end
                    end
                end
            end
        end
    end
    toc
end

save('Projection','obs_proj');



function para = checkbound_para(para,paramax,paramin)
for i=1:size(para,1)
    para(i,para(i,:)<paramin(i))=paramin(i)*(1+0.1*rand(sum(para(i,:)<paramin(i)),1));
    para(i,para(i,:)>paramax(i))=paramax(i)*(1-0.1*rand(sum(para(i,:)>paramax(i)),1));
end

function [S,E,Ir,Iu]=checkbound(S,E,Ir,Iu,C)
for k=1:size(S,2)
    S(S(:,k)<0,k)=0; E(E(:,k)<0,k)=0; Ir(Ir(:,k)<0,k)=0; Iu(Iu(:,k)<0,k)=0;
end


function [S,E,Ir,Iu]=adjustmobility(S,E,Ir,Iu,nl,part,MI_inter_relative,t)
num_loc=size(MI_inter_relative,1);
for l=1:num_loc
    for j=part(l)+1:part(l+1)-1
        if t<=size(MI_inter_relative,2)
            S(part(l))=S(part(l))+((1-MI_inter_relative(nl(j),t))*S(j));
            S(j)=(MI_inter_relative(nl(j),t)*S(j));
            E(part(l))=E(part(l))+((1-MI_inter_relative(nl(j),t))*E(j));
            E(j)=(MI_inter_relative(nl(j),t)*E(j));
            Ir(part(l))=Ir(part(l))+((1-MI_inter_relative(nl(j),t))*Ir(j));
            Ir(j)=(MI_inter_relative(nl(j),t)*Ir(j));
            Iu(part(l))=Iu(part(l))+((1-MI_inter_relative(nl(j),t))*Iu(j));
            Iu(j)=(MI_inter_relative(nl(j),t)*Iu(j));
        end
    end
end

function [C1,Cave1,eva_ori,eva_dest,NE,SE,EE,IEr,IEu,S,E,Ir,Iu]=getC_eva(C,Cave,nl,part,S,E,Ir,Iu,V,evaids,destids)
num_ens=size(S,2);
C1=C;
Cave1=Cave;
%set up eva_ori,eva_dest
num_eva=sum(sum(V>0));
eva_ori=zeros(num_eva,1);
eva_dest=zeros(num_eva,1);
NE=zeros(num_eva,1);
SE=zeros(num_eva,num_ens);EE=SE;IEr=SE;IEu=SE;
cnt=0;
for i=1:length(evaids)
    evaid=evaids(i);
    for j=1:length(destids)
        if V(j,i)>0
            cnt=cnt+1;
            destid=destids(j);
            eva_ori(cnt)=evaid;
            eva_dest(cnt)=destid;
            NE(cnt,:)=V(j,i);
            %set up new subpopulations
            SE(cnt,:)=(V(j,i)*sum(S(part(evaid):part(evaid+1)-1,:))/sum(C(part(evaid):part(evaid+1)-1)));
            EE(cnt,:)=(V(j,i)*sum(E(part(evaid):part(evaid+1)-1,:))/sum(C(part(evaid):part(evaid+1)-1)));
            IEr(cnt,:)=(V(j,i)*sum(Ir(part(evaid):part(evaid+1)-1,:))/sum(C(part(evaid):part(evaid+1)-1)));
            IEu(cnt,:)=(V(j,i)*sum(Iu(part(evaid):part(evaid+1)-1,:))/sum(C(part(evaid):part(evaid+1)-1)));
        end
    end
end

for i=1:length(evaids)
    evaid=evaids(i);
    %update evaucated subpopulations
    gamma=1-sum(V(:,i))/sum(C(part(evaid):part(evaid+1)-1));
    C1(part(evaid):part(evaid+1)-1)=(C(part(evaid):part(evaid+1)-1)*gamma);
    Cave1(part(evaid):part(evaid+1)-1)=0;%residence
    Cave1(nl==evaid)=0;%work
    S(part(evaid):part(evaid+1)-1,:)=(S(part(evaid):part(evaid+1)-1,:)*gamma);
    E(part(evaid):part(evaid+1)-1,:)=(E(part(evaid):part(evaid+1)-1,:)*gamma);
    Ir(part(evaid):part(evaid+1)-1,:)=(Ir(part(evaid):part(evaid+1)-1,:)*gamma);
    Iu(part(evaid):part(evaid+1)-1,:)=(Iu(part(evaid):part(evaid+1)-1,:)*gamma);
end


function [S,E,Ir,Iu]=returnhome(C,part,S,E,Ir,Iu,eva_ori,SE,EE,IEr,IEu)

for i=1:length(eva_ori)
    ori=eva_ori(i);
    for j=part(ori):part(ori+1)-1
        S(j,:)=S(j,:)+SE(i,:)*C(j)/sum(C(part(ori):part(ori+1)-1));
        E(j,:)=E(j,:)+EE(i,:)*C(j)/sum(C(part(ori):part(ori+1)-1));
        Ir(j,:)=Ir(j,:)+IEr(i,:)*C(j)/sum(C(part(ori):part(ori+1)-1));
        Iu(j,:)=Iu(j,:)+IEu(i,:)*C(j)/sum(C(part(ori):part(ori+1)-1));
    end
end