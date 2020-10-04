function [Gs,op,nodes] = my_mk_nbrs_of_dag(G0,simple,CLASS)
% MK_NBRS_OF_DAG Make all DAGs that differ from G0 by a single edge deletion, addition or reversal
% [Gs, op, nodes] = mk_nbrs_of_dag(G0)
%
% Gs{i} is the i'th neighbor.
% op{i} = 'add', 'del', or 'rev' is the operation used to create the i'th neighbor.
% nodes(i,1:2) are the head and tail of the operated-on arc.
% simple is optional (default 0). for simple=1 no arc reversal is allowed
% CLASS is optional (defaults 0). for CLASS~=0 use Markov Blanket property

%default for simple is 0, default for CLASS is 0
if nargin<2
    simple=0;
    CLASS=0;
elseif nargin==2
    CLASS=0;
end


Gs = {};
op = {};
nodes = [];

if CLASS==0
    [I,J] = find(G0);
    nnbrs = 1;
    % all single edge deletions
    for e=1:length(I)
        i = I(e); j = J(e);
        G = G0;
        G(i,j) = 0;
        Gs{nnbrs} = G;
        op{nnbrs} = 'del';
        nodes(nnbrs, :) = [i j];
        nnbrs = nnbrs + 1;
    end
    
    % all single edge reversals
    if simple==0
        for e=1:length(I)
            i = I(e); j = J(e);
            G = G0;
            G(i,j) = 0;
            G(j,i) = 1;
            if acyclic(G)
                Gs{nnbrs} = G;
                op{nnbrs} = 'rev';
                nodes(nnbrs, :) = [i j];
                nnbrs = nnbrs + 1;
            end
        end
    end
    
    [I,J] = find(~G0);
    % all single edge additions
    for e=1:length(I)
        i = I(e); j = J(e);
        if i ~= j % don't add self arcs
            G = G0;
            G(i,j) = 1;
            if G(j,i)==0 % don't add i->j if j->i exists already
                if acyclic(G)
                    Gs{nnbrs} = G;
                    op{nnbrs} = 'add';
                    nodes(nnbrs, :) = [i j];
                    nnbrs = nnbrs + 1;
                end
            end
        end
    end
else
    current_mb_dag=zeros(size(G0));
    current_mb_dag(CLASS,:)=G0(CLASS,:);
    current_mb_dag(:,[CLASS find(G0(CLASS,:))])=G0(:,[CLASS find(G0(CLASS,:))]);
    [I,J] = find(G0);
    nnbrs = 1;
    % all single edge deletions
    for e=1:length(I)
        i = I(e); j = J(e);
        G = G0;
        G(i,j) = 0;
        mb_dag=zeros(size(G));
        mb_dag(CLASS,:)=G(CLASS,:);
        mb_dag(:,[CLASS find(G(CLASS,:))])=G(:,[CLASS find(G(CLASS,:))]);
        if ~isequal(mb_dag,current_mb_dag)
            Gs{nnbrs} = G;
            op{nnbrs} = 'del';
            nodes(nnbrs, :) = [i j];
            nnbrs = nnbrs + 1;
        end
    end
    
    % all single edge reversals
    if simple==0
        for e=1:length(I)
            i = I(e); j = J(e);
            G = G0;
            G(i,j) = 0;
            G(j,i) = 1;
            if acyclic(G)
                mb_dag=zeros(size(G));
                mb_dag(CLASS,:)=G(CLASS,:);
                mb_dag(:,[CLASS find(G(CLASS,:))])=G(:,[CLASS find(G(CLASS,:))]);
                if ~isequal(mb_dag,current_mb_dag)
                    Gs{nnbrs} = G;
                    op{nnbrs} = 'rev';
                    nodes(nnbrs, :) = [i j];
                    nnbrs = nnbrs + 1;
                end
            end
        end
    end
    
    [I,J] = find(~G0);
    % all single edge additions
    for e=1:length(I)
        i = I(e); j = J(e);
        if i ~= j % don't add self arcs
            G = G0;
            G(i,j) = 1;
            if G(j,i)==0 % don't add i->j if j->i exists already
                if acyclic(G)
                    mb_dag=zeros(size(G));
                    mb_dag(CLASS,:)=G(CLASS,:);
                    mb_dag(:,[CLASS find(G(CLASS,:))])=G(:,[CLASS find(G(CLASS,:))]);
                    if ~isequal(mb_dag,current_mb_dag)
                        Gs{nnbrs} = G;
                        op{nnbrs} = 'add';
                        nodes(nnbrs, :) = [i j];
                        nnbrs = nnbrs + 1;
                    end
                end
            end
        end
    end
end








