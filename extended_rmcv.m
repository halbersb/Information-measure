function [results,mats] = extended_rmcv(mytrain_data,mytest_data,CLASS,num_folds_k,init_type,score_type,varargin)
    % RMCV based on 'score_type'. An extention for the RMCV.
    % CLASS       - integer - the index of the target variable
    % num_folds_k - integer - the number of fold for cross-validation
    % init_type   - string  - the type of initialization
    % score_type  - string  - the score metric
    % varargin    - relevant only to IMalpha or to synthetic problems
    % written by Dan Halbersberg (halbsers@post.bgu.ac.il)
    
    %define the score metric to be used
    flag=0;
    switch score_type,
        case 'ACC',     func=str2func('calculate_ACC');flag=1; %flag means to do (1-ACC) so we can minimize
        case 'CEN',     func=str2func('calculate_CEN');
        case 'F1',      func=str2func('calculate_F');
        case 'IM',      func=str2func('calculate_IM');
        case 'IMalpha', func=str2func('calculate_IMalpha');
        case 'MAE',     func=str2func('calculate_MAE');
        case 'MCC',     func=str2func('calculate_MCC');flag=1; %flag means to do (1-MCC) so we can minimize
        case 'MI',      func=str2func('calculate_MI');
        otherwise, error('invalid input for score_type');       
    end

    mytrain_data=cell2mat(mytrain_data.');
    mytest_data=cell2mat(mytest_data.');

    %find sizes
    [num_train_samples,num_attributes]=size(mytrain_data);
    num_test_samples=size(mytest_data,1);
    nodes_sizes=max(mytrain_data);
    num_classes=nodes_sizes(CLASS);
    switch length(varargin),
        case 0, alpha=num_classes;
        case 1, if size(varargin{1},1)==1, alpha=varargin{1}; else Real_MB=varargin{1}; alpha=num_classes; end
        case 2, alpha=varargin{1}; Real_MB=varargin{2};
    end

    %make k folds validation sets
    fold_size=floor(num_train_samples/num_folds_k);
    train_folds=cell(num_folds_k,1);
    for c1=1:num_folds_k-1
        train_folds{c1}=mytrain_data(1+fold_size*(c1-1):fold_size*c1,:);
    end
    train_folds{num_folds_k}=mytrain_data(1+fold_size*c1:end,:);

    %make k training sets
    train_sets=cell(num_folds_k,1);
    for c1=1:num_folds_k-1
        train_sets{c1}=mytrain_data;
        train_sets{c1}(1+fold_size*(c1-1):fold_size*c1,:)=[];
    end
    train_sets{num_folds_k}=mytrain_data;
    train_sets{num_folds_k}(1+fold_size*c1:end,:)=[];

    %make initial graph (DIBN, NBC or Empty)
    init_dag=zeros(num_attributes,num_attributes);
    if strcmp(init_type,'D'), init_dag(:,CLASS)=1; end
    if strcmp(init_type,'NBC'), init_dag(CLASS,:)=1; end %else empty graph
    init_dag(CLASS,CLASS)=0;
    dag_nodes_parents=cell(num_attributes,1);
    parents_index=1:num_attributes;
    for c1=1:num_attributes
        dag_nodes_parents{c1}=parents_index(logical(init_dag(:,c1)));
    end

    %make initial conditional probability tables
    train_folds_sizes=zeros(num_folds_k,1);
    train_sets_sizes=zeros(num_folds_k,1);
    prob_tables=cell(num_folds_k,num_classes);
    for c1=1:num_folds_k
        train_folds_sizes(c1)=size(train_folds{c1},1);
        train_sets_sizes(c1)=size(train_sets{c1},1);
        for c2=1:num_classes
            prob_tables{c1,c2}=zeros(train_folds_sizes(c1),num_attributes);
        end
    end
    for c1=1:num_folds_k
        current_sub_train_sets=cell(num_attributes,1);
        for current_attribute=1:num_attributes
            current_sub_train_sets{current_attribute}=train_sets{c1}(:,[current_attribute dag_nodes_parents{current_attribute}]);
        end
        for current_sample_index=1:train_folds_sizes(c1)
            current_sample=train_folds{c1}(current_sample_index,:);
            for c2=1:num_classes
                current_sample(CLASS)=c2;
                for current_attribute=1:num_attributes
                    current_sub_sample=current_sample([current_attribute dag_nodes_parents{current_attribute}]);
                    rep_current_sub_sample=current_sub_sample(ones(train_sets_sizes(c1),1),:);
                    current_compare=current_sub_train_sets{current_attribute}==rep_current_sub_sample;
                    current_compare=current_compare(logical(all(current_compare(:,2:end),2)),:);
                    current_compare_parents_sum=size(current_compare,1);
                    current_compare_sum=sum(current_compare(:,1));
                    prob_tables{c1,c2}(current_sample_index,current_attribute)=current_compare_sum/current_compare_parents_sum;
                end
            end
        end
    end
    confusion_matrix=zeros(num_classes,num_classes);
    prod_prob_tables=cell(num_folds_k,1);
    for c1=1:num_folds_k
        prod_prob_tables{c1}=zeros(train_folds_sizes(c1),num_classes);
        for c2=1:num_classes
            prod_prob_tables{c1}(:,c2)=prod(prob_tables{c1,c2},2);
        end
        [~,I]=max(prod_prob_tables{c1},[],2);
        for c3=1:size(I,1)
            confusion_matrix(I(c3),train_folds{c1}(c3,CLASS))=confusion_matrix(I(c3),train_folds{c1}(c3,CLASS))+1;
        end
    end
    if strcmp(score_type,'IMalpha')
        current_score=func(confusion_matrix,num_train_samples,alpha);
    else
        current_score=func(confusion_matrix,num_train_samples);
        if flag, current_score=1-abs(current_score); end
    end
    current_confusion_matrix=confusion_matrix;
    current_dag=init_dag;
    current_prob_tables=prob_tables;
    next_score=current_score;
    next_confusion_matrix=current_confusion_matrix;
    nbrs_counter=0;
    iteration_counter=0;
    iteration_flag=1;
    %% start the search and score procedure
    while iteration_flag
        current_score=next_score;
        current_confusion_matrix=next_confusion_matrix;
        [dag_nbrs,dag_nbrs_op,dag_nbrs_nodes]=my_mk_nbrs_of_dag(current_dag,0,CLASS);
        num_nbrs=size(dag_nbrs,2);
        nbrs_counter=nbrs_counter+num_nbrs;
        iteration_counter=iteration_counter+1;
        for c3=1:num_nbrs
            if strcmp(dag_nbrs_op{c3},'del') || strcmp(dag_nbrs_op{c3},'add')
                affected_nodes=dag_nbrs_nodes(c3,2);
            else
                affected_nodes=dag_nbrs_nodes(c3,:);
            end
            dag_nodes_parents=cell(num_attributes,1);
            for c2=1:num_attributes
                dag_nodes_parents{c2}=parents_index(logical(dag_nbrs{c3}(:,c2)));
            end
            tested_prob_tables=current_prob_tables;
            for c1=1:num_folds_k
                tested_sub_train_sets=current_sub_train_sets;
                for current_attribute=affected_nodes
                    tested_sub_train_sets{current_attribute}=train_sets{c1}(:,[current_attribute dag_nodes_parents{current_attribute}]);
                end
                for current_sample_index=1:train_folds_sizes(c1)
                    current_sample=train_folds{c1}(current_sample_index,:);
                    for c2=1:num_classes
                        current_sample(CLASS)=c2;
                        for current_attribute=affected_nodes
                            current_sub_sample=current_sample([current_attribute dag_nodes_parents{current_attribute}]);
                            rep_current_sub_sample=current_sub_sample(ones(train_sets_sizes(c1),1),:);
                            current_compare=tested_sub_train_sets{current_attribute}==rep_current_sub_sample;
                            current_compare=current_compare(logical(all(current_compare(:,2:end),2)),:);
                            current_compare_parents_sum=size(current_compare,1);
                            current_compare_sum=sum(current_compare(:,1));
                            tested_prob_tables{c1,c2}(current_sample_index,current_attribute)=current_compare_sum/current_compare_parents_sum;
                        end
                    end
                end
            end
            tested_prod_prob_tables=cell(num_folds_k,1);
            confusion_matrix=zeros(num_classes,num_classes);
            for c1=1:num_folds_k
                tested_prod_prob_tables{c1}=zeros(train_folds_sizes(c1),num_classes);
                for c2=1:num_classes
                    tested_prod_prob_tables{c1}(:,c2)=prod(tested_prob_tables{c1,c2},2);
                end
                [~,I]=max(tested_prod_prob_tables{c1},[],2);
                for c4=1:size(I,1)
                    confusion_matrix(I(c4),train_folds{c1}(c4,CLASS))=confusion_matrix(I(c4),train_folds{c1}(c4,CLASS))+1;
                end
            end
            if strcmp(score_type,'IMalpha')
                validation_score=func(confusion_matrix,num_train_samples,alpha);
            else
                validation_score=func(confusion_matrix,num_train_samples);
                if flag, validation_score=1-abs(validation_score); end
            end
            %if a better neighbor was found
            if validation_score<next_score
                next_score=validation_score;
                next_confusion_matrix=confusion_matrix;
                next_dag=dag_nbrs{c3};
                next_prob_tables=tested_prob_tables;
                next_sub_train_sets=tested_sub_train_sets;
            end
        end
        %if the best neighbor is better than the current graph
        if next_score<current_score
            current_score=next_score;
            current_confusion_matrix=next_confusion_matrix;
            current_dag=next_dag;
            current_prob_tables=next_prob_tables;
            current_sub_train_sets=next_sub_train_sets;
        else
            iteration_flag=0;
        end
    end
    nodes_sizes=max([nodes_sizes;max(mytest_data)]);
    %% calculate all performence measure for the selected model
    [~,tested_mat,P]     = rmcv_score(current_dag,nodes_sizes,mytrain_data,mytest_data,CLASS);
    tested_ACC_score     = calculate_ACC(tested_mat,num_test_samples);
    tested_CEN_score     = calculate_CEN(tested_mat,num_test_samples);
    tested_F1_score      = calculate_F1(tested_mat,num_test_samples);
    tested_IM_score      = calculate_IM(tested_mat,num_test_samples);
    tested_IMalpha_score = calculate_IMalpha(tested_mat,num_test_samples,alpha); %alpha=num_classes for non IMalpha runs
    tested_MAE_score     = calculate_MAE(tested_mat,num_test_samples);
    tested_MAE_w_score   = calculate_MAE_w(tested_mat,num_test_samples);
    tested_MCC_score     = calculate_MCC(tested_mat,num_test_samples);
    tested_MI_score      = calculate_MI(tested_mat,num_test_samples);
    tested_AUC_score     = calculate_AUC(P,mytest_data(:,CLASS));
    if exist('Real_MB','var') %for synthetic  problems
        SHD_score=SHD(Real_MB,select_markov_blanket(current_dag,CLASS));
        results=[tested_ACC_score,tested_CEN_score,tested_F1_score,tested_IM_score,tested_IMalpha_score,tested_MAE_score,tested_MAE_w_score,tested_MCC_score,tested_MI_score,tested_AUC_score,nbrs_counter,iteration_counter,SHD_score];
    else
        results=[tested_ACC_score,tested_CEN_score,tested_F1_score,tested_IM_score,tested_IMalpha_score,tested_MAE_score,tested_MAE_w_score,tested_MCC_score,tested_MI_score,tested_AUC_score,nbrs_counter,iteration_counter];
    end
    mats{1}=current_confusion_matrix;
    mats{2}=tested_mat;
    mats{3}=current_dag;
end

%% all score functions start from here
function [score,mat,prod_prob_tables] = rmcv_score(dag,nodes_sizes,train_data,test_data,CLASS)
    %returns accuracy, confusion_matrix, and posteriori probabilities for AUC
    %written by Roi Kelner modified by Dan Halbersberg

    num_classes=nodes_sizes(CLASS);
    [num_train_samples,num_attributes]=size(train_data);
    num_test_samples=size(test_data,1);
    dag_nodes_parents=cell(num_attributes,1);
    parents_index=1:num_attributes;
    for c1=1:num_attributes
        dag_nodes_parents{c1}=parents_index(logical(dag(:,c1)));
    end
    prob_tables=cell(1,num_classes);

    for c1=1:num_classes
        prob_tables{1,c1}=zeros(num_test_samples,num_attributes);
    end

    current_sub_train_sets=cell(num_attributes,1);
    for current_attribute=1:num_attributes
        current_sub_train_sets{current_attribute}=train_data(:,[current_attribute dag_nodes_parents{current_attribute}]);
    end
    for current_sample_index=1:num_test_samples
        current_sample=test_data(current_sample_index,:);
        for c1=1:num_classes
            current_sample(CLASS)=c1;
            for current_attribute=1:num_attributes
                current_sub_sample=current_sample([current_attribute dag_nodes_parents{current_attribute}]);
                rep_current_sub_sample=current_sub_sample(ones(num_train_samples,1),:);
                current_compare=current_sub_train_sets{current_attribute}==rep_current_sub_sample;
                current_compare=current_compare(logical(all(current_compare(:,2:end),2)),:);
                current_compare_parents_sum=size(current_compare,1);
                current_compare_sum=sum(current_compare(:,1));
                prob_tables{1,c1}(current_sample_index,current_attribute)=current_compare_sum/current_compare_parents_sum;
            end
        end
    end
    mat=zeros(num_classes,num_classes);
    prod_prob_tables=zeros(num_test_samples,num_classes);
    for c1=1:num_classes
        prod_prob_tables(:,c1)=prod(prob_tables{1,c1},2);
    end
    [~,I]=max(prod_prob_tables,[],2);
    %create confusion matrix
    for c2=1:size(I,1)
        mat(I(c2),test_data(c2,CLASS))=mat(I(c2),test_data(c2,CLASS))+1;
    end
    %calculate accuracy
    score=sum(I==test_data(:,CLASS));
    score=score/num_test_samples;
end

%%
function [score] = calculate_ACC(confusion_matrix,total_cases)
    %accuracy - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    true=sum(diag(confusion_matrix));
    score=true/total_cases;
end

%%
function [score] = calculate_CEN(confusion_matrix,total_cases)
    %CEN - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    score=0;
    N=size(confusion_matrix,1)-1;
    for j=1:size(confusion_matrix,1)
        Pj=(sum(confusion_matrix(:,j))+sum(confusion_matrix(j,:)))/(2*total_cases); 
        CENj=0;
        Pkj=0;
        Pjk=0;
        for k=1:size(confusion_matrix,2)
            if j~=k && (sum(confusion_matrix(j,:))+sum(confusion_matrix(:,j)))>0
                Pkj=confusion_matrix(k,j)/(sum(confusion_matrix(j,:))+sum(confusion_matrix(:,j)));
                Pjk=confusion_matrix(j,k)/(sum(confusion_matrix(j,:))+sum(confusion_matrix(:,j)));
                %only non-zero cells
                if Pkj*Pjk>0
                    CENj=CENj-Pjk*logb(Pjk,2*N)-Pkj*logb(Pkj,2*N);
                else
                    %only non-zero cells
                    if Pjk>0
                        CENj=CENj-Pjk*logb(Pjk,2*N);
                    end
                    %only non-zero cells
                    if Pkj>0
                        CENj=CENj-Pkj*logb(Pkj,2*N);
                    end
                end
            end
        end    
        score=score+Pj*CENj;    
    end
end

%%
function [score] = calculate_IM(confusion_matrix,total_cases)
    %IM - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    score=0;
    for i=1:size(confusion_matrix,1)
        for j=1:size(confusion_matrix,2)
            Pxy=confusion_matrix(i,j)/total_cases;
            Px=sum(confusion_matrix(i,:))/total_cases;
            Py=sum(confusion_matrix(:,j))/total_cases;
            if Pxy>0
                score=score+(Pxy*(-log2(Pxy/(Px*Py))+log2(1+abs(i-j))));
            end
        end
    end
end

%%
function [score] = calculate_IMalpha(confusion_matrix,total_cases,alpha)
    %IMalpha - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix
    %"alpha" is the trade-off parameter

    score=0;
    for i=1:size(confusion_matrix,1)
        for j=1:size(confusion_matrix,2)
            Pxy=confusion_matrix(i,j)/total_cases;
            Px=sum(confusion_matrix(i,:))/total_cases;
            Py=sum(confusion_matrix(:,j))/total_cases;
            if Pxy>0
                if abs(i-j)>0 %if x<>y
                    score=score+(Pxy*(-log2(alpha*Pxy/(Px*Py))+log2(alpha*(1+abs(i-j)))));
                else
                    score=score+(Pxy*(-log2(alpha*Pxy/(Px*Py))+log2(1+abs(i-j))));
                end
            end
        end
    end
end

%%
function [score] = calculate_MAE(confusion_matrix,total_cases)
    %MAE - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    score=0;
    for i=1:size(confusion_matrix,1)
        for j=1:size(confusion_matrix,2)
            score=score+confusion_matrix(i,j)*abs(j-i);
        end
    end
    score=score/total_cases;
end

%%
function [score] = calculate_MCC(confusion_matrix,total_cases)
    %MCC - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    score=0;
    N=size(confusion_matrix,1);
    X=zeros(total_cases,N);
    Y=zeros(total_cases,N);
    %build to metrices (SxN) where S in number of samples and N is number of variables
    temp=1;
    for i=1:N
        for j=1:N
            if temp<(confusion_matrix(i,j)+temp-1)
                for k=temp:(confusion_matrix(i,j)+temp-1)
                    X(k,i)=1;
                    Y(k,j)=1;
                end
                temp=k+1;
            end
        end
    end
    %calculate Xi avg and Yi avy
    Xavg=zeros(1,N);
    Yavg=zeros(1,N);
    for i=1:N
        for s=1:total_cases
            Xavg(1,i)=Xavg(1,i)+X(s,i);
            Yavg(1,i)=Yavg(1,i)+Y(s,i);
        end
        Xavg(1,i)=Xavg(1,i)/total_cases;
        Yavg(1,i)=Yavg(1,i)/total_cases;
    end
    %calculate cov(x,y)
    covXY=0;
    for s=1:total_cases
        for i=1:N
            covXY=covXY+((X(s,i)-Xavg(1,i))*(Y(s,i)-Yavg(1,i)));
        end
    end
    covXY=covXY/N;
    %calculate cov(x,x)
    covXX=0;
    for s=1:total_cases
        for i=1:N
            covXX=covXX+((X(s,i)-Xavg(1,i))*(X(s,i)-Xavg(1,i)));
        end
    end
    covXX=covXX/N;
    %calculate cov(y,y)
    covYY=0;
    for s=1:total_cases
        for i=1:N
            covYY=covYY+((Y(s,i)-Yavg(1,i))*(Y(s,i)-Yavg(1,i)));
        end
    end
    covYY=covYY/N;

    if covXX*covYY>0
        score=covXY/(power((covXX*covYY),0.5));
    else
        score=0;
    end
end

%%
function [score] = calculate_MI(confusion_matrix,total_cases)
    %MI - measure for confusion matrix
    %"confusion_matrix" is a matrix
    %"total_cases" is the sum of all entries in the confusion_matrix

    score=0;
    for i=1:size(confusion_matrix,1)
        for j=1:size(confusion_matrix,2)
            Pxy=confusion_matrix(i,j)/total_cases;
            Px=sum(confusion_matrix(i,:))/total_cases;
            Py=sum(confusion_matrix(:,j))/total_cases;
            %only non-zero cells
            if Pxy>0
                score=score+(Pxy*(-log2(Pxy/(Px*Py))));
            end
        end
    end
end

%%
function [score] = calculate_F1(confusion_matrix,total_cases)
	%F1 - measure for confusion matrix
	%"confusion_matrix" is the matrix
	%"total_cases" is the sum of all entries in confusion_matrix

	score=0;
	N=size(confusion_matrix,1);
	precision=0;
	recall=0;
	%option a (F1)
	%for i=1:N
	%    tpi=confusion_matrix(i,i);
	%    fpi=sum(confusion_matrix(:,i));
	%    fni=sum(confusion_matrix(i,:));
	%    if fpi>0, precision=precision+tpi/(fpi); end
	%    if fni>0, recall=recall+tpi/(fni); end
	%end

	%precision=precision/N;
	%recall=recall/N;
	%score=(2*precision*recall)/(precision+recall);

	%option b (macro-F1)
	for i=1:N
		tpi=confusion_matrix(i,i);
		fpi=sum(confusion_matrix(:,i));
		fni=sum(confusion_matrix(i,:));
		precision=tpi/fpi;
		recall=tpi/fni;
		if isnan((2*precision*recall)/(precision+recall))
			score(i)=0;
		else
			score(i)=(2*precision*recall)/(precision+recall);
		end
	end
	score=mean(score);
end

%%
function [score] = calculate_MAE_w(confusion_matrix,total_cases)
	%MAE with weights - measure for confusion matrix (for imbalnce data according to Baccianella et al., 2009)
	%"confusion_matrix" is the matrix
	%"total_cases" is the sum of all entries in confusion_matrix

	score=0;
	for i=1:size(confusion_matrix,1)
		class_score=0;
		weigth=sum(confusion_matrix(:,i));
		if weigth>0
			for j=1:size(confusion_matrix,2)
				class_score=class_score+confusion_matrix(j,i)*abs(j-i);
			end
			score=score+class_score/weigth;
		end
	end
	score=score/size(confusion_matrix,1);
end

%%
function mAUC = calculate_AUC(P,label)
	%this function calculates AUC for binary or multiclass problems (hand and till, 2001)
	%[P] is a matrix with probabilty for all classes of target variable -
	%    row = num of samples in test set
	%    col = num of classes of target variavle
	%[label] is an array of target variable values

    %initialization
    [num_cases,num_classes]=size(P);
    comb=nchoosek(1:num_classes,2);
    if num_classes-length(unique(label))>1
        %remove comb of non existing (both) classes in test set
        dif=setdiff(1:num_classes,unique(label));
        comb2=nchoosek(dif,2);
        for i=1:size(comb2,1)
            comb(find(comb(:,1)==comb2(i,1) & comb(:,2)==comb2(i,2)),:)=[];
        end
    end
    P(isnan(P))=0; %replace nans with zeros
    for i=1:num_cases, if sum(P(i,:))<1, P(i,:)=P(i,:)/sum(P(i,:)); end; end %if P's row does not sumup to 1 then normalize it
    mAUC=0;
    
    %for each pair of classes calculate A(Ci,Cj)
    for i=1:size(comb,1)
        mAUC=mAUC+(a_value(P,label,comb(i,1),comb(i,2))+a_value(P,label,comb(i,2),comb(i,1)))/2;
    end
    
    mAUC=mAUC*(2/(num_classes*(num_classes-1)));
    if mAUC<0.5, mAUC=1-mAUC; end
end

%%
function val = a_value(P,label,zero_label,one_label)
    expanded_points=[];
    cnt=1;
    for i=1:size(P,1)
        if label(i)==zero_label || label(i)==one_label
            expanded_points(cnt,:)=[label(i) P(i,zero_label)];
            cnt=cnt+1;
        end
    end
    sorted_ranks=sortrows(expanded_points,2);
    n0=0;n1=0;sum_ranks=0;
    for i=1:size(sorted_ranks,1)
        if sorted_ranks(i,1)==zero_label, n0=n0+1; sum_ranks=sum_ranks+i; end
        if sorted_ranks(i,1)==one_label, n1=n1+1; end
    end
    if sum_ranks==(n0*(n0+1)/2) && n0*n1==0 %both are zero
        val=1; %shouldn't be zero???
    else
        val=(sum_ranks-(n0*(n0+1)/2))/(n0*n1);
    end
end