function test()

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
num_loc=size(part,1)-1;
num_mp=size(nl,1);

load dailyincidence
load dailydeaths

num_times=size(dailyincidence,2);%total length of data
%smooth the data: 7 day moving average
for l=1:num_loc
    for t=1:num_times
        dailyincidence(l,t)=mean(dailyincidence(l,max(1,t-3):min(t+3,num_times)));
        dailydeaths(l,t)=mean(dailydeaths(l,max(1,t-3):min(t+3,num_times)));
    end
end

load deathrate_IFR
T=50;
Teva=30;%evacuate on day 30
obs_case=dailyincidence;
obs_death=dailydeaths;
incidence=dailyincidence;%to initialize

%adjusting inter-county movement
load MI_inter
%adjusting mobility starting from March 16, day 25
MI_inter_relative=MI_inter(:,2:end);
for t=25:size(MI_inter_relative,2)
    MI_inter_relative(:,t)=MI_inter_relative(:,t)./MI_inter_relative(:,t-1);
    MI_inter_relative(isnan(MI_inter_relative(:,t)),t)=0;
    MI_inter_relative(isinf(MI_inter_relative(:,t)),t)=0;
    MI_inter_relative(:,t)=min(MI_inter_relative(:,t),1);
end
MI_inter_relative(:,1:24)=1;
%intercounty-mobility does not change after evacuation
MI_inter_relative(:,Teva:end)=1;


C=C*ones(1,num_times);
Cave=Cave*ones(1,num_times);
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

num_ens=5;%number of ensemble
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run model unitl March 10, day 19
[S,E,Ir,Iu,Seedc]=initialize(nl,part,C(:,1),num_ens,incidence);
obs_temp=zeros(num_loc,num_ens,T);%records of reported cases
death_temp=zeros(num_loc,num_ens,T);%records of death
load('parafit1')
%initialize parameters
[para,paramax,paramin,betamap,alphamaps]=initializepara_eakf(dailyincidence,num_ens,parafit);
%%%%%%%%%%%%inflate variables and parameters
lambda=2;
para=mean(para,2)*ones(1,num_ens)+lambda*(para-mean(para,2)*ones(1,num_ens));
para=checkbound_para(para,paramax,paramin);
E=mean(E,2)*ones(1,num_ens)+lambda*(E-mean(E,2)*ones(1,num_ens));
Ir=mean(Ir,2)*ones(1,num_ens)+lambda*(Ir-mean(Ir,2)*ones(1,num_ens));
Iu=mean(Iu,2)*ones(1,num_ens)+lambda*(Iu-mean(Iu,2)*ones(1,num_ens));
[S,E,Ir,Iu]=checkbound(S,E,Ir,Iu,C(:,1));
%fix Z, D and theta
%Z
para(1,:)=parafit(3,1:num_ens);
%D
para(2,:)=parafit(4,1:num_ens);
%mu
para(3,:)=parafit(2,1:num_ens);
%theta
para(4,:)=parafit(6,1:num_ens);

for t=1:Teva
    t
    tic
    %%%%%%%%%%%seeding
    if t<=size(Seedc,2)
        [S,E,Ir,Iu]=seeding(S,E,Ir,Iu,nl,part,C(:,t),Seedc,t);
    end
    for k=1:num_ens
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k)]=adjustmobility(S(:,k),E(:,k),Ir(:,k),Iu(:,k),nl,part,MI_inter_relative,t);
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,dailyIu_temp]=model_eakf(nl,part,C(:,t),Cave(:,t),S(:,k),E(:,k),Ir(:,k),Iu(:,k),para(:,k),betamap,alphamaps);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
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
        %death delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=round((dailyIr_temp(j)+dailyIu_temp(j))*deathrate(l,min(t,size(deathrate,2))));
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
    Stotal(t)=mean(sum(S));
    toc
end

load evacuation
%set evacuation subpopulation
[C1,Cave1,eva_ori,eva_dest,NE,SE,EE,IEr,IEu,S,E,Ir,Iu]=getC_eva(C(:,Teva),Cave(:,Teva),nl,part,S,E,Ir,Iu,V,evaids,destids);

num_mpE=size(NE,1);

L_eva=4;%evacuation for 4 days

for t=Teva+1:Teva+L_eva
    t
    tic
    for k=1:num_ens
        %no need to adjust cross-county mobility
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,dailyIu_temp,SE(:,k),EE(:,k),IEr(:,k),IEu(:,k),dailyIEr_temp,dailyIEu_temp]=...
            model_eva(nl,part,C1,Cave1,S(:,k),E(:,k),Ir(:,k),Iu(:,k),para(:,k),betamap,alphamaps,eva_ori,eva_dest,NE,SE(:,k),EE(:,k),IEr(:,k),IEu(:,k),evaids,destids);
        
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
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
        %death delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=round((dailyIr_temp(j)+dailyIu_temp(j))*deathrate(l,min(t,size(deathrate,2))));
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
        %add evacuees to resident counties
        %reporting delay
        for i=1:num_mpE
            l=eva_ori(i);
            inci=dailyIEr_temp(i);
            if inci>0
                rnd=datasample(rnds,inci);
                for h=1:length(rnd)
                    if (t+rnd(h)<=T)
                        obs_temp(l,k,t+rnd(h))=obs_temp(l,k,t+rnd(h))+1;
                    end
                end
            end
        end
        %death delay
        for i=1:num_mpE
            l=eva_ori(i);
            inci=round((dailyIEr_temp(i)+dailyIEu_temp(i))*deathrate(l,min(t,size(deathrate,2))));
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
    
    Stotal(t)=mean(sum(S))+mean(sum(SE));
    toc
end

%return
[S,E,Ir,Iu]=returnhome(C(:,Teva+L_eva),part,S,E,Ir,Iu,eva_ori,SE,EE,IEr,IEu);

for t=Teva+L_eva+1:T
    t
    tic
    for k=1:num_ens
        [S(:,k),E(:,k),Ir(:,k),Iu(:,k),dailyIr_temp,dailyIu_temp]=model_eakf(nl,part,C(:,t),Cave(:,t),S(:,k),E(:,k),Ir(:,k),Iu(:,k),para(:,k),betamap,alphamaps);
        %reporting delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=dailyIr_temp(j);
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
        %death delay
        for l=1:num_loc
            for j=part(l):part(l+1)-1
                inci=round((dailyIr_temp(j)+dailyIu_temp(j))*deathrate(l,min(t,size(deathrate,2))));
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
    Stotal(t)=mean(sum(S));
    toc
end


% figure(1)
% plot(1:T,sum(squeeze(mean(obs_temp,2))))
% figure(2)
% plot(1:T,Stotal)



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



