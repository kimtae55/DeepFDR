##### MT functions

bh.func<-function(pv, q)
{ 
  # the input 
    # pv: the p-values
    # q: the FDR level
  # the output 
    # nr: the number of hypothesis to be rejected
    # th: the p-value threshold
    # de: the decision rule

  m=length(pv)
  st.pv<-sort(pv)   
  pvi<-st.pv/1:m
  de<-rep(0, m)
  if (sum(pvi<=q/m)==0)
  {
    k<-0
    pk<-1
  }
  else
  {
    k<-max(which(pvi<=(q/m)))
    pk<-st.pv[k]
    de[which(pv<=pk)]<-1
  }
  y<-list(nr=k, th=pk, de=de)
  return (y)
}

por.func<-function(pv, pii, q)
{ 
  # the input 
    # pv: the p-values
    # q: the FDR level
  # the output 
    # nr: the number of hypothesis to be rejected
    # th: the p-value threshold
    # de: the decision rule

  qq<-q/(1-pii)
  m=length(pv)
  st.pv<-sort(pv)   
  pvi<-st.pv/1:m
  de<-rep(0, m)
  if (sum(pvi<=qq/m)==0)
  {
    k<-0
    pk<-1
  }
  else
  {
    k<-max(which(pvi<=(qq/m)))
    pk<-st.pv[k]
    de[which(pv<=pk)]<-1
  }
  y<-list(nr=k, th=pk, de=de)
  return (y)
}

sab.func<-function(pvs, pis, q)
{
    ## implementing "SABHA" by Li and Barber
	## Arguments
	 # pvs: p-values
	 # pis: conditional probabilities
	 # q: FDR level
	## Values
	 # de: the decision
	 # th: the threshold for weighted p-values
	 
  m<-length(pvs)
  nu<-10e-5
  pis[which(pis>1-nu)]<-1-nu # stabilization
  pws<-pvs*(1-pis)
  st.pws<-sort(pws)

  pwi<-st.pws/1:m
  de<-rep(0, m)
  if (sum(pwi<=q/m)==0)
  {
    k<-0
    pk<-1
  }
  else
  {
    k<-max(which(pwi<=(q/m)))
    pk<-st.pws[k]
    de[which(pws<=pk)]<-1
  }
  y<-list(nr=k, th=pk, de=de)
  return (y)
}	 

law.func<-function(pvs, pis, q)
{
    ## implementing "spatial multiple testing by locally adaptive weighting"
	## Arguments
	 # pvs: p-values
	 # pis: conditional probabilities
	 # q: FDR level
	## Values
	 # de: the decision
	 # th: the threshold for weighted p-values
	 
	m<-length(pvs)
	nu<-10e-5
	pis[which(pis<nu)]<-nu # stabilization
	pis[which(pis>1-nu)]<-1-nu # stabilization
	ws<-pis/(1-pis)
    pws<-pvs/ws
    st.pws<-sort(pws)
    fdps<-sum(pis)*st.pws/(1:m)
    de<-rep(0, m)
    if(sum(fdps<=q)==0)
    {
    	k<-0
    	pwk<-1
    }
    else
    {
    	k<-max(which(fdps<=q))
    	pwk<-st.pws[k]
    	de[which(pws<=pwk)]<-1
    }
    y<-list(nr=k, th=pwk, de=de)
	return (y)
}	 

epsest.func <- function(x,u,sigma)
{
  # x is a vector of normal variables
  # u is the mean 
  # sigma is the standard deviation
  # the output is the estimated non-null proportion
  
  z  = (x - u)/sigma
  xi = c(0:100)/100
  tmax=sqrt(log(length(x)))
  tt=seq(0,tmax,0.1)

  epsest=NULL

  for (j in 1:length(tt)) { 

    t=tt[j]
    f  = t*xi
    f  = exp(f^2/2)
    w  = (1 - abs(xi))
    co  = 0*xi

    for (i in 1:101) {
      co[i] = mean(cos(t*xi[i]*z));
    } 
    epshat = 1 - sum(w*f*co)/sum(w)
    epsest=c(epsest,epshat)
  }
  return(epsest=max(epsest))
}

scr.func<-function(pv, eps, alpha)
{  
	## the function constructs a subset with 100*alpha% signals
	 # Reference: Section 3.1 in Jin (2008)
	## Arguments 
	 # pv is a vector of p-values
	 # eps is the estimate of the non-null proportion
	 # alpha is the screening level 
	## Values
	 # de: the decision
	 # th: the threshold for screening
    
  m=length(pv)
  de<-rep(0, m)
  st.pv<-sort(pv)   
  ntp.vec<-(1:m)-(1-eps)*m*st.pv-alpha*m*eps
  if (sum(ntp.vec>=0)==0)
  {
  	k<-0
  	pk<-1
  }
  else
  {
    k<-min(which(ntp.vec>=0))
    pk<-st.pv[k]
    de[which(pv<=pk)]<-1
  }
  y<-list(nr=k, th=pk, de=de)
  return (y)	
}

pis.lin.func<-function(h, x1, x2)
{
	len<-x2-x1
	y<-rep(0, len)
	a_u<-2*h/len
	b_u<--2*h*x1/len
	a_d<--2*h/len
	b_d<-2*h*x2/len
	
	for(i in 1:len)
	{
		if(i<len/2)
		y[i]<-a_u*(x1+i)+b_u
		else
		y[i]<-a_d*(x1+i)+b_d
	}
	return(y)
}


lin.itp<-function(x, X, Y){
  ## x: the coordinates of points where the density needs to be interpolated
  ## X: the coordinates of the estimated densities
  ## Y: the values of the estimated densities
  ## the output is the interpolated densities
  x.N<-length(x)
  X.N<-length(X)
  y<-rep(0, x.N)
  for (k in 1:x.N){
    i<-max(which((x[k]-X)>=0))
    if (i<X.N)
      y[k]<-Y[i]+(Y[i+1]-Y[i])/(X[i+1]-X[i])*(x[k]-X[i])
    else 
      y[k]<-Y[i]
  }
  return(y)
}

pis_1D.func2<- function(t_1, t_2, method=c('lfdr','pval'), tau=0.1, bdw=200)
{
  ## pis_est.func calculates the conditional proportions pis
  ## Arguments
   # t_1: the primary statistic
   # t_2: the auxiliary variable (taken as the location in a spatial domain in this project)
   # tau: the screening threshold, which can be prespecified or chosen adaptively
  ## Values
   # pis: the conditional proportions

  m <- length(t_1)
  # Calculate the p-values
  t_1.pval <- 2*pnorm(-abs(t_1))

  # Use Jin and Cai's method for global non-null proportion estimation
  t_1.p.est.global <- epsest.func(t_1,0,1)
  p.est <-rep(0,m)

  t_1.den.est.init <- density(t_1,from=min(t_1)-10,to=max(t_1)+10,n=1000)
  t_1.den.est <- lin.itp(t_1,t_1.den.est.init$x,t_1.den.est.init$y)

  t_2.den.est.init <- density(t_2, bw=bdw, from=min(t_2)-10,to=max(t_2)+10,n=1000)
  t_2.den.est <- lin.itp(t_2,t_2.den.est.init$x,t_2.den.est.init$y)

  if(method=='lfdr'){###Screening with lfdr
    #Calculate lfdr test statistics and truncate at 1
    t_1.lfdr.est <- (1-t_1.p.est.global)*dnorm(t_1)/t_1.den.est
    t_1.lfdr.est[which(t_1.lfdr.est>1)] <- 1

    #empirically calculate the correction factor
    sample.null <- rnorm(1000)
    sample.lfdr <- (1-t_1.p.est.global)*dnorm(sample.null)/lin.itp(sample.null,t_1.den.est.init$x,t_1.den.est.init$y)
    correction <- length(which(sample.lfdr>=tau))/1000

    T.tau <- which(t_1.lfdr.est>=tau)
    t_2.screened.den.est <- density(t_2[T.tau], bw=bdw, from=min(t_2[T.tau])-10,to=max(t_2[T.tau])+10,n=1000)
    t_2.screened.den.est <- lin.itp(t_2,t_2.screened.den.est$x,t_2.screened.den.est$y)

    p.est <- length(T.tau)/m*t_2.screened.den.est/t_2.den.est/correction
    p.est[which(p.est>1)] <- 1
  }

  if(method=='pval'){ ###Screening with p-values
    T.tau <- which(t_1.pval>=tau)
    t_2.screened.den.est <- density(t_2[T.tau], bw=bdw, from=min(t_2[T.tau])-10, to=max(t_2[T.tau])+10, n=1000)
    t_2.screened.den.est <- lin.itp(t_2,t_2.screened.den.est$x,t_2.screened.den.est$y)

    p.est <-length(T.tau)/m*t_2.screened.den.est/t_2.den.est/(1-tau)
    p.est[which(p.est>1)] <-1
  }
  return(1-p.est)
}

pis_1D.func<- function(x, tau=0.1, h=50)
{
  ## pis_est.func calculates the conditional proportions pis
  ## Arguments
   # x: z-values
   # tau: the screening threshold, which can be prespecified or chosen adaptively
   # bdw: bandwidth
  ## Values
   # pis: the conditional proportions

  m <- length(x)
  s <- 1:m # auxiliary variable
  pval <- 2*pnorm(-abs(x))
  p.est <-rep(0, m)
  for (i in 1:m) { 
  	kht<-dnorm(s-i, 0, h)
  	p.est[i]<-sum(kht[which(pval>=tau)])/((1-tau)*sum(kht))
  }
  p.est[which(p.est>1)] <-1
  return(1-p.est)
}

disvec.func<-function(dims, s)
{
   # disvec computes the distances of all points on a m1 times m2 spatial domain to a point s
  ## Arguments:
   # dims=c(d1, d2): the dimensions
   # s=c(s1, s2): a spatial point
  ## Values:
   # a vector of distances
   
   m<-dims[1]*dims[2]
   dis.vec<-rep(0, m)
   for(i in 1:dims[1])
   {
   	dis.vec[((i-1)*dims[2]+1):(i*dims[2])]<-sqrt((i-s[1])^2+(1:dims[2]-s[2])^2)
   }
   return(dis.vec) 
}

pis_2D.func<- function(x, tau=0.1, h=10)
{
  ## pis_2D.func calculates the conditional proportions pis
  ## Arguments
   # x: a matrix of z-values
   # tau: the screening threshold, which can be prespecified or chosen adaptively
   # bdw: bandwidth
  ## Values
   # pis: conditional proportions

  dims<-dim(x)
  m<-dims[1]*dims[2]
  x.vec<-c(t(x))
  pv.vec<-2*pnorm(-abs(x.vec))
  scr.idx<-which(pv.vec>=tau)
  p.est<-matrix(rep(0, m), dims[1], dims[2])  
  
  for (i in 1:dims[1]) 
  {
  	for (j in 1:dims[2]) 
  	{
  		s<-c(i, j)
  		dis.vec<-disvec.func(dims, s)
  		kht<-dnorm(dis.vec, 0, h)
    	p.est[i,j]<-min(1-1e-5, sum(kht[scr.idx])/((1-tau)*sum(kht)))
  	}
  }
  pis.est<-1-c(p.est)
  return(pis.est)
}

disvec.func.3D<-function(dims, s)
{
   # disvec computes the distances of all points on a m1 times m2 spatial domain to a point s
  ## Arguments:
   # dims=c(d1, d2, d3): the dimensions
   # s=c(s1, s2, s3): a spatial point
  ## Values:
   # a vector of distances

   m<-dims[1]*dims[2]*dims[3]
   dis.vec<-rep(0, m)
   loc<-0
   for(k in 1:dims[3])
   {
   	 for (j in 1:dims[2])
   	 {
   	 	for(i in 1:dims[1])
   	 	{
   	 		loc<-loc+1
   	 		dis.vec[loc]<-sqrt((i-s[1])^2+(j-s[2])^2+(k-s[3])^2)
   	 	}
   	 }
   }
   return(dis.vec) 
}

pis_3D.func<- function(x, tau=0.1, h=5)
{
  ## pis_2D.func calculates the conditional proportions pis
  ## Arguments
   # x: a matrix of z-values
   # tau: the screening threshold, which can be prespecified or chosen adaptively
   # bdw: bandwidth
  ## Values
   # pis: conditional proportions

  dims<-dim(x)
  m<-dims[1]*dims[2]*dims[3]
  x.vec<-c(x)
  pv.vec<-2*pnorm(-abs(x.vec))
  scr.idx<-which(pv.vec>=tau)
  p.est<-array(rep(0, m), dims)  
  
  for (i in 1:dims[1]) 
  {
  	for (j in 1:dims[2])
  	{
  		for (k in 1:dims[3])
  		{
	  		s<-c(i,j,k)
	  		dis.vec<-disvec.func.3D(dims, s)
	  		kht<-dnorm(dis.vec, 0, h)
	    	p.est[i,j,k]<-min(1-1e-5, sum(kht[scr.idx])/((1-tau)*sum(kht)))
  	    }

  	} 
  }
  pis.est<-1-c(p.est)
  return(pis.est)
}


