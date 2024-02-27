close all
clear all


A = readtable('all_signals.csv');

t = A{:,2};
s(:,1:5) = A{:,3:7};

fext = [2 4 5 6 8];


fe = 50;% Hz
Nf = 2048;
f = [0:Nf-1]*fe/Nf;

%[PKS,LOCS] =findpeaks(s);
%[PKS2,LOCS2] =findpeaks(-s);

indt= 5:55;%5:55;

val = NaN(length(fext),1);
freq_rep = NaN(length(fext),1);

figure
for nn = 1:5

    %%%% look for the regress line = low freq tendency
[coeff,bint,r,rint,stats] = regress(s(indt,nn),[t(indt) ones(length(indt) ,1)]);

%%% remove the regress line
s2 = zeros(length(t),1);

delta = s(indt(1),nn)-(t(indt(1))*coeff(1)+coeff(2));

s2(indt) = s(indt,nn)-(t(indt)*coeff(1)+coeff(2));%-delta;

s3 = zeros(Nf,1);
s3(indt) = s(indt,nn)-(t(indt)*coeff(1)+coeff(2));


[PKS,LOCS] =findpeaks(s2);
[PKS2,LOCS2] =findpeaks(-s2);


subplot(2,1,1)   %%%%% plot in time
plot(t,s(:,nn))
hold on;plot(t(indt),t(indt)*coeff(1)+coeff(2),'b--')
hold on;plot(t,s2,'k')
%hold on; plot(t(LOCS),PKS,'ok')
%hold on; plot(t(LOCS2),-PKS2,'or')
hold off

S = fft(s(:,nn)-mean(s(:,nn)),Nf); %%% TF original signal
S2 = fft(s2(indt)-mean(s2(indt)),Nf); %%%% TF signal - line 

S3 = fft(s3-mean(s2)); %%%% TF signal - line 

subplot(2,1,2)   %%%%% plot in freq
%plot(f,abs(S));hold on;
plot(f,abs(S2),'k')
hold on;plot(f,abs(S3),':')

if nn==1
    [a b] = max(abs(S2));  %%%%% find the max pos + values
    val(nn)=a;
    freq_rep(nn)=f(b);
else 
    ind0 = max(find(f<=freq_rep(nn-1)*1.2));
    indmax = max(find(f<=10));
    [a b] = max(abs(S2(ind0:indmax)));

    val(nn)=a;  
    freq_rep(nn)=f(b+ind0-1);
end

xlim([0 10])
hold on;plot(fext(nn)*[1 1],[0 val(nn)],'r-','LineWidth',3)
hold on;plot(freq_rep(nn),val(nn),'ok','LineWidth',3)



hold off
title(nn)

pause

end


figure(10);
subplot(1,2,1)
hold on
plot(freq_rep(1:5),val(1:5),'ok-')
subplot(1,2,2)
hold on
plot(fext(1:5),freq_rep(1:5),'ok-')
hold on;plot(fext(1:5),fext(1:5),'k--')
grid on
axis('equal')
axis([1 9 1 9])

%fmax = 10;
%indf = find(f<= fmax);

%filter = zeros(Nf,1);
%filter(indf)= tukeywin(length(indf),0.3);
