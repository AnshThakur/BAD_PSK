function [h,gama_nk] = calc_h(data,m,v,w)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUTHORS               : Dileep A.D. 
% DATE                  : 01/05/2010
% LAST MODIFIED         : 14/07/2015
% OTHER FUNCTIONS USED  : 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    A=data;
    t=size(data,1);
    nbmix=size(m,1);
   
    % ---- Compute the responsibility term ----
    [nbdata,dim]=size(A);
    for mixNum=1:nbmix
        sg=1./v(mixNum,:);
        sig=diag(sg); clear sg;
	term1=-(1/2)*sum(log(v(mixNum,:)));  
% This change is made iorder to avoid det going to 0 (i.e. prod(vv(mixNum,:)))o 
% If this change is not made, it will make det 0, which makes the constant term Inf. 
% This make likelihood go NaN
% Idea here is apply log to N(.) i.e. normal distribution of a component
        term2=-(dim/2)*log(2*pi);
%         diff=(A-repmat(m(mixNum,:),nbdata,1));
%         mdist=diff*sig*diff'; 
%         maha_dist=diag(mdist); 
        for i=1:nbdata
            maha_dist(i)=(A(i,:)-m(mixNum,:))*sig*(A(i,:)-m(mixNum,:))';
        end
	term3=-0.5*maha_dist; 
        term=term1+term2+term3; % compute the log likelihood for a component
        nr_tmp=w(mixNum)*exp(term);	  
% Aplly exp(.) to get likelihood. Then multiply with mixture weight. 
        nr(:,mixNum)=nr_tmp;
    end
    nr=nr';
    s=sum(nr);
    gama_nk=nr./repmat(s,nbmix,1);clear nr; 
    gama_nk=gama_nk'; % gama_nk=[nbdata,nbmix];
    % -----------------------------------------
    h=(1/t) .* sum(gama_nk);  % Effective number of examples for each mixture, normalized by number of examples; [1,nbmix]
    

    
    
    
    
    
